## Metircs
This folder contains the code of calculating the metrics for reference. It should be noted that our models MMedLM, MMedLM 2, and MMed-Llama3 are all base models and have not been fine-tuned. For evaluation, we fine-tuned all the open source models tested (as well as our MMedLM series base models) on the MMedBench trainset. After fine-tuning, the models can be output in a fixed format, which is convenient for comparison with the standard answers.

The following are the scripts for calculating the Accuracy, BLEU, ROUGE, and Bert-score metrics:
- Accuracy: Calculated by comparing the predicted value with the actual value. See metrics_accuracy.py for details
- BLEU: You can use the nltk.translate.bleu_score module in the NLTK library. See rationale_similarity.py for details
- ROUGE: You can use the rouge library to calculate. See rationale_similarity.py for details
- Bert-score: Calculated using the bert-score library. See rationale_similarity.py for details.