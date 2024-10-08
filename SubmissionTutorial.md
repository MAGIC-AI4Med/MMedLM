# MMedBench Submission Tutorial Guidelines

Welcome to the MMedBench Leaderboard Submission Guidelines! MMedBench utilizes two evaluation methods: one for multiple-choice accuracy (ACC), and another for rationale similarity (BLEU, ROUGE). Note that the multiple-choice metric is calculated across the full MMedBench-test dataset, while the rationale similarity is only evaluated on selected cases from the MMedBench-test that have passed human-checking.

In this tutorial, we will guide you through the necessary steps to submit your model and results for official evaluation on the MMedBench. 

## Submission Methods
Please include the following in your submission:

1. **Model Description**: Provide a brief description of your model, including:
    - Model Name
    - Model Size
    - References (e.g., Hugging Face URL, relevant papers)
    - Training details (e.g., pretraining on medical corpora, finetuning on medical datasets or the MMedBench trainset, use of RHLF, application of in-context learning)

2. **Results File**: Submit CSV files containing your model's output results:
   - `multi_choice_results.csv`: This file should contain four columns: "Language", "ModelInput", "ModelOutput", and "GT_Answer", where "GT_Answer" indicates the correct choice(s), such as "A" or "A,B,C".
   - `rationale_results.csv`: This file should also contain four columns "Language", "ModelInput", "ModelOutput", and "GT_Answer". It includes only cases that passed human verification. "GT_Answer" here refers to the golden rationale from the MMedBench-test.

3. **Evaluation Results**: Submit CSV files with your model's evaluation metrics, which can be referenced in [metrics foler](https://github.com/MAGIC-AI4Med/MMedLM/tree/main/metrics):
   - `accuracy_metric.csv`: Displays accuracy across different languages with columns for each of the six languages.
   - `rationale_similarity_metric.csv`: Shows rationale similarity across different languages and metrics (BLEU-1 to BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L), with the first column indicating the metric and the remaining columns for each language.

Please email the required documents to Pengcheng Qiu at qiupengcheng@pjlab.org.cn with the subject line "[MMedBench Submission] - Name_of_model".

## Removal from the Leaderboard
If you wish to have your model's scores removed from the leaderboard, please contact us via email. Removals will be processed during the next leaderboard update.