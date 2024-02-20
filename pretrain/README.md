## Auto-regressive training on MMedC

### step 1 prepare the dataset
In order to reduce the time overhead caused by CPU processing data during training, we choose to uniformly perform data preprocessing first. All corpus is divided according to tcontextext length and tokenized in this step. The tokenized chunk is finally saved in a folder in `.npy`` format. This step is a CPU intensive task. In our experiment, it may take 1-2 hours with 16 CPUs.

Before running the code, you should first prepare `TRAIN_FILENAMES_TXT` with paths to all the `.txt` format corpus, and set `SAVE_DIR` to save the `.npy` processed data files.
```bash
# For MMedLM:
sbatch MMedLM_datapreparing.sh

# Or for MMedLM2:
sbatch MMedLM2_datapreparing.sh
```

### step 2 start auto-regressive training
when all the data are prepared, we can start auto-regressive training, which injects medical knowledge into the LLM.  Notice that this step requires considerable computing resources. In our experiment, we trained the model on 8 A100 for over 30 Days.


Before running the code, you should set `TRAIN_DATASET_FOLDER`, which is the folder save `.npy` files in previous steps, and set `OUTPUT_DIR`, which is the folder to save the checkpoint.
```bash
# For MMedLM:
sbatch MMedLM_pretraining.sh

# Or for MMedLM2:
sbatch MMedLM2_pretraining.sh
```