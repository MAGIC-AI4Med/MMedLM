## Fine-tuning on MMedBench Trainset
we provide the code for further fine-tuning on the MMedBench Trainset, along with all the hyperparameters used in our experiments. 
We employed two fine-tuning methods, namely Full model fine-tuning and PEFT fine-tuning.
### Full Model Fine-tuning
Full model fine-tuning typically enables the model to achieve better results, but due to the 7B LLM being too large to fit on a single A100 80GB GPU, it is necessary to use FSDP (Fully Sharded Data Parallel) technology to distribute the model across multiple GPUs. In our experiments, we used 4 A100 80GB GPUs.

| Method     |  global_batch_size     | learning_rate         | fsdp_transformer_layer_cls_to_wrap|
|------------|------------------------|-----------------------|-----------------------------------|
| BLOOMZ     | 128                    |1e-6                   |BloomBlock|
| InternLM   | 128                    |1e-6                   |InternLMDecoderLayer|
| Llama 2    | 128                    |1e-6                   |LlamaDecoderLayer|
| MedAlpaca  | 128                    |1e-6                   |LlamaDecoderLayer|  
| ChatDoctor | 128                    |1e-6                   |LlamaDecoderLayer|
| PMC-LLaMA  | 128                    |1e-6                   |LlamaDecoderLayer|
| Mistral    | 128                    |1e-6                   |MistralDecoderLayer|
| InternLM 2 | 128                    |1e-6                   |InternLM2DecoderLayer|
| MMedLM     | 128                    |1e-6                   |InternLMDecoderLayer|
| MMedLM 2   | 128                    |1e-6                   |InternLM2DecoderLayer|

For full model finetuning, you should download the model weight weights first, and change the following parameters in `fullmodel_finetuning.sh`:
- MODEL_NAME_OR_PATH
- OUTPUT_DIR: Directory to save the checkpoints
- TRANSFORMER_LAYER: fsdp_transformer_layer_cls_to_wrap, used for FSDP.

Then you can start full model finetuning by 
```bash
sbatch fullmodel_finetuning.sh
```
### PEFT Fine-tuning (LoRA)
PEFT fine-tuning is generally more efficient (faster speed, less memory usage) for fine-tuning models because it freezes most of the parameters of the LLM and only updates a small number of parameters. Fine-tuning the model using LoRA allows the model to be placed on a single A100 80GB GPU. In our experiments, we employed the DDP (Distributed Data Parallel) parallel strategy, using 4 A100 80GB GPUs.

| Method     |  global_batch_size     | learning_rate         | target_modules                    |
|------------|------------------------|-----------------------|-----------------------------------|
| BLOOMZ     | 128                    |1e-6                   |["query_key_value"]  |
| InternLM   | 128                    |1e-6                   |["q_proj", "v_proj"] |
| Llama 2    | 128                    |1e-6                   |["q_proj", "v_proj"] |
| MedAlpaca  | 128                    |1e-6                   |["q_proj", "v_proj"] |  
| ChatDoctor | 128                    |1e-6                   |["q_proj", "v_proj"] |
| PMC-LLaMA  | 128                    |1e-6                   |["q_proj", "v_proj"] |
| Mistral    | 128                    |1e-6                   |["q_proj", "v_proj"] |
| InternLM 2 | 128                    |1e-6                   |["wqkv"]             |
| MMedLM     | 128                    |1e-6                   |["q_proj", "v_proj"] |
| MMedLM 2   | 128                    |1e-6                   |["wqkv"] |

For full model finetuning, you should download the model weight weights first, and change the following parameters in `fullmodel_finetuning.sh`:
- MODEL_NAME_OR_PATH
- OUTPUT_DIR: Directory to save the checkpoints
- TARGET_MODUILES: target_modules, used for LoRA.

Then you can start LoRA finetuning by 
```bash
sbatch LoRA_finetuning.sh
```