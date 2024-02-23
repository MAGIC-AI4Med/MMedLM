## Inference on MMedBench Testset

We provide the code for inference on MMedBench Testset. Note that currently MMedLM is a multilingual medical foundation model that hasn't undergone Supervised Fine-Tuning. You may need to first finetune the model with code in the **finetune** folder.

After finetuning, you need to change the following parameters in `inference.sh` before running it.
- MODEL_NAME_OR_PATH : The folder containing the fintuned checkpoints of the model.
- IS_LORA: If the model is fine-tuned with LoRA. If `True`, we will use `AutoPeftModel` to load the weight.
- OUTPUT_DIR: The directory to save the response of the model.
- IS_WITH_RATIONALE: If the model need to output the rationale of it's choice. If `True`, the prompt which instructs the model to output both selected option and rationale is applied. If `False`, model will output only the selected option, which is easier to calculate Accuracy.


Then you can start inference by 
```bash
sbatch inference.sh
```