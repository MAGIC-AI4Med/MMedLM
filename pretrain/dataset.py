import os
import copy
import transformers
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing import Pool
from random import shuffle

seed = 42
transformers.set_seed(seed)

class InternlmDataset(Dataset):
    def __init__(self, data_dir:str, low_mem=True, num_processes=4) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        shuffle(self.filenames)
        self.data = None
        if low_mem == False:
            print("Loading the dataset into memory...")
            with Pool(num_processes) as pool:
                self.data = list(tqdm(pool.imap(self.load_npy, self.filenames), total=len(self.filenames)))

    def __len__(self):
        return len(self.filenames)
    
    def load_npy(self, filename):
        npy_path = self.data_dir + '/' + filename
        return np.load(npy_path)
    
    def __getitem__(self, idx):
        if self.data is not None:
            input_id = self.data[idx]
        else:
            input_id = self.load_npy(self.filenames[idx])
        label = copy.deepcopy(input_id)
        return dict(input_ids=input_id, labels=label)
    