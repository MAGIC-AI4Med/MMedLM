import os
import json
import copy
import torch
import logging
import transformers
from random import shuffle
from transformers import Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List


SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100
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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_cache : bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    gradient_clipping : float = field(
        default=None
    )
    
    
def jsonl_load(data_path):
    """Load a .jsonl file into a dictionary."""
    filepaths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.jsonl')]    
    src_dict_ls = []
    for filepath in filepaths:
        lang = os.path.basename(filepath).split(".")[0]
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                src_dict = json.loads(line)
                src_dict["lang"] = lang
                src_dict_ls.append(src_dict)
                        
    res_dict_ls = []
    for src_dict in src_dict_ls:
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
        data_with_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step. You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Rationale: {rationale}\n###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_with_rationale)
        
        data_without_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_without_rationale) 
               
    # shuffle the data for training
    shuffle(res_dict_ls)                
    return res_dict_ls

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jsonl_load(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )
    
    if model_args.is_lora:
        if model_args.target_modules is None:
            raise ValueError("target_modules is required for LoRA")
        
        config = LoraConfig(
            r = model_args.lora_rank,
            lora_alpha = model_args.lora_alpha,
            target_modules = model_args.target_modules,
            lora_dropout = 0.05,
            bias = 'none',
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, config)
            
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
