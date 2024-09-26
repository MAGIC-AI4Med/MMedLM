"""
@Description :   calculate the accuracy of the model
@Author      :   Henrychur 
@Time        :   2024/09/13 10:20:40
"""
filepaths = [
    "./USMLE_en_res.txt",
    "./USMLE_zh_res.txt",
    "./lgakuQA_res.txt",
    "./FrenchMedMCQA_res.txt",
    "./RuMedDaNet_res.txt",
    "./Head-QA_res.txt"
]

for filepath in filepaths:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_cnt = len(lines)
        right_cnt = 0
        error_cnt = 0
        for line in lines:
            line = line.strip()
            response, gt_answer, gt_rationale = line.split("[SPLIT]")[0], line.split("[SPLIT]")[1], line.split("[SPLIT]")[2]
            response = response.replace("###Answer: OPTION ", "").replace("IS CORRECT.</s>", "").strip()
            if len(gt_answer) == 1:
                if response == gt_answer:
                    right_cnt += 1
            else:
                gt_answer = gt_answer.split(",")
                response = response.split(",")
                if set(response) == set(gt_answer):
                    right_cnt += 1
        print("{}\t{}\t{}\t{:.2f}".format(filepath, right_cnt, total_cnt, right_cnt / total_cnt * 100))
