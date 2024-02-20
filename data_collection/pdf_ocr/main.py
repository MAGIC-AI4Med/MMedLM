from PDFParser import PDFParser
import os
import tqdm
import json

PDF_ROOT = "./data/ebooks_pdf"
SAVE_ROOT = "./pdf2txt/ebooks_txt"


def pdf2json(pdf_parser, pdf_path, txt_path):
    '''
        Convert PDF to JSON
    '''
    text_ls = pdf_parser.extract_pdf(pdf_path)
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))
    # write to json file
    text_json = {
        "pdf_filename": pdf_path.split("/")[-1],
        "page_num": len(text_ls),
        'text_list': text_ls
    }
    with open(txt_path.replace(".txt", ".json"), 'w') as f:
        json.dump(text_json, f, indent=4, ensure_ascii=False)


def get_all_pdf_path(pdf_root):
    '''
        Get all pdf path recursively
    '''
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_root):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    print("len(pdf_paths): {}".format(len(pdf_paths)))
    return pdf_paths


def write_path():
    '''
        Write all pdf paths to 8 txt files
    '''
    pdf_paths = get_all_pdf_path(PDF_ROOT)
    # split into 8 parts
    pdf_paths = [pdf_paths[i::8] for i in range(8)]
    # write to 8 txt for multiprocessing
    for i in range(8):
        with open("paths/"+str(i)+".txt", 'w') as f:
            for path in pdf_paths[i]:
                f.write(path+"\n")


def inference(mode):
    '''
        Convert pdf to json with paddleocr.
        'mode' is the index of the txt file in the paths folder, this function can be used for multiprocessing
    '''
    assert mode in [0, 1, 2, 3, 4, 5, 6, 7]
    paths = []
    with open("paths/"+str(mode)+".txt", 'r') as f:
        for line in f:
            line = line.strip()
            # open the result txt, if the txt's length is less than 5000, add it to list
            txt_path = line.replace(
                ".pdf", ".json").replace(PDF_ROOT, SAVE_ROOT)
            if os.path.exists(txt_path):
                continue
            paths.append(line)

    print("mode: {}, len(paths): {}".format(mode, len(paths)))
    pdf_parser = PDFParser()
    for pdf_path in tqdm.tqdm(paths):
        try:
            pdf2json(pdf_parser, pdf_path, SAVE_ROOT +
                    pdf_path.replace(PDF_ROOT, "").replace('.pdf', '.txt'))
        except:
            with open("error.txt", 'a') as f:
                f.write(pdf_path+"\n")
                f.flush()


if __name__ == "__main__":
    # split pdf paths into 8 parts
    # which can be processed by 8 gpus (if you have 8 gpus)
    if not os.path.exists("paths"):
        os.makedirs("paths")
        write_path()

    # if you want to use muti-gpus, and you should replace the 'for' loop with multi-processing
    for i in range(8):
        inference(i)
