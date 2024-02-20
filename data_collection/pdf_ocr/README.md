# Tool for converting PDF ebooks to txt
## Installation
To get started,  you should install the necessary dependencies for performing OCR (Optical Character Recognition) on PDFs. We've chosen to use **paddleocr** due to its superior performance in both Chinese and English OCR tasks.

If you haven't already, install PaddlePaddle by running:

```bash
conda install paddlepaddle-gpu==2.5.1 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```
If you have other CUDA version, please refer to https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html

After installing PaddlePaddle, proceed to install the **paddleocr** library with a minimum version of 2.0.1:

```bash
pip install "paddleocr==2.7.0"
```

Next, we need to install some other dependencies:

```bash
pip install -r requirements.txt
```

## Some problems you may encounter
1. paddleocr's dependencies may not be installed correctly. See issue on paddleocr repo: https://github.com/PaddlePaddle/PaddleOCR/issues/8914
```bash
pip install PyMuPDF==1.18.0
```

2. Sometimes paddle won't be able to find the correct CUDNN path. You can try to set the path manually:
```bash 
export LD_LIBRARY_PATH=/root/anaconda3/envs/paddle/lib:$LD_LIBRARY_PATH
```

## Work Flow
After successfully setting up the environment and making the required adjustments to `SAVE_ROOT` and `PDF_ROOT`, run `python main.py`. The program will initially scan and collect all PDF files located within the `PDF_ROOT` directory, distributing them into eight equally sized segments and storing them in eight separate text files. This approach streamlines parallel processing for enhanced efficiency. Following this, the default code employs a serial for loop to execute the process on a single GPU.
```python
for i in range(8):
    inference(i)
```

Here is an approach to implement multiprocessing, which can be conveniently specified using argparse to determine the input mode. It then initiates a process to execute it, making it particularly convenient for use on a Slurm cluster.
```python
import argparse
parser = argparse.ArgumentParser(description="Run inference with different modes")
parser.add_argument("mode", type=int, help="Mode to run (0-7)")

args = parser.parse_args()
mode = args.mode
inference(mode)
```