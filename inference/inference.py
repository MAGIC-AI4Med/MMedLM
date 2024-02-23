import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import Sequence
from peft import AutoPeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--is_with_rationale', type=bool, default=False)
    parser.add_argument('--is_lora', type=bool, default=False)
    args = parser.parse_args()
    return args


def inference_on_one(input_str: Sequence[str], model, tokenizer) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        model_inputs.input_ids.cuda(),
        max_new_tokens=1000,
        top_k=50
    )
    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]



def read_jsonl(filepath: str, is_with_rationale):
    """Load a .jsonl file into a dictionary."""
    src_dict_ls = []
    lang = os.path.basename(filepath).split(".")[0]
    with open(filepath, "r") as f:
        for line in f:
            src_dict = json.loads(line)
            src_dict["lang"] = lang
            src_dict_ls.append(src_dict)
            
    res_dict_ls = []
    for src_dict in src_dict_ls:
        question = src_dict["question"]
        lang = src_dict["lang"]
        question = src_dict["question"]
        options = ""
        for key in src_dict["options"].keys():
            content = src_dict["options"][key]
            options += f"{key}. {content} "
        if isinstance(src_dict["answer_idx"], str):
            answer_id = src_dict["answer_idx"]
        elif isinstance(src_dict["answer_idx"], list):
            answer_id = ",".join(src_dict["answer_idx"])
            
        rationale = src_dict["rationale"]
        if is_with_rationale:
            tmp = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step.  You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
                "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output":f"{answer_id}",
                "rationale":f"{rationale}"
            }    
        else:
            tmp = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
                "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
                "output":f"{answer_id}",
                "rationale":f"{rationale}"
            }    
        res_dict_ls.append(tmp)
        
    return res_dict_ls


def prepare_data(data_list: Sequence[dict]) -> Sequence[dict]:
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        data_list[_idx]['sample_id'] = _idx

        data_list[_idx]['pmc_input'] = prompt_input.format_map(data_entry) if data_entry.get("input", "") != "" else prompt_no_input.format_map(data_entry)
        data_list[_idx]['pmc_output'] = data_entry['output']
        data_list[_idx]['rationale'] = data_entry['rationale']
    return data_list

def inference(test_filepath, model, tokenizer, save_dir, is_with_rationale):
    data_list = read_jsonl(test_filepath, is_with_rationale)
    data_list = prepare_data(data_list, model, tokenizer)
    answers = []
    for _idx in tqdm(range(len(data_list))):
        data_entry = data_list[_idx]
        sample_id = data_entry['sample_id']
        input_str = [
            data_entry['pmc_input']
        ]
        output_str = inference_on_one(input_str)
        response = output_str.split("### Response:")[1].strip()
        answers.append((response.replace("\n", ""), data_entry['pmc_output'], data_entry['rationale']))
        
    with open(save_dir + "/" +test_filepath.split("/")[1]+"_res.txt", "w", encoding="utf-8") as fp:
        for response, target, rationale in answers:
            fp.write(f"{response}[SPLIT]{target}[SPLIT]{rationale}\n")
            
def validate():
    args = parse_args()
    filepaths = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path) if filename.endswith('.jsonl')]  
    if args.is_lora == False:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
    else:
        model = AutoPeftModel.from_pretrained(args.model_name_or_path)
    
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=2048,
        use_fast=False,
        trust_remote_code=True
    )

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
        
    for idx, filepath in enumerate(filepaths):
        inference(filepath, model, tokenizer, args.save_dir, args.is_with_rationale)
        print(idx, "\t", filepath, " has done!")
    
if __name__ == '__main__':
    validate()
