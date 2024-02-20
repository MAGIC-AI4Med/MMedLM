import os
import zipfile
import json

def find_tuples_recursive(lst):
    tuples = []
    for item in lst:
        if isinstance(item, tuple):
            tuples.append(item)
        elif isinstance(item, list):
            tuples.extend(find_tuples_recursive(item))
    return tuples

def cal_ocr_token_nums():
    # calculate the number of tokens in the Chinese ocr txt
    tokens = 0
    cnt = 0
    for i in range(8):
        txt_paths = []
        with open("paths/{}.txt".format(i), 'r') as f:
            for line in f:
                line = line.replace(".pdf", ".txt")
                line = line.replace("/remote-home/share/medical/public/multilingual/Medical Books/", "/remote-home/pengchengqiu/projects/multilingual-data/demo_data/")
                txt_paths.append(line.strip())
        for path in txt_paths:
            if os.path.exists(path) == False:
                continue
            with open(path, "r", encoding="utf-8") as fp:
                text = fp.read()
                cleaned_text = ''.join(filter(lambda char: char.isalpha(), text))
            tokens += len(cleaned_text)
            cnt+=1
    print(tokens)
    print(cnt)

def cal_crawl_token_nums():
    # calculate the number of tokens in the crawled json file
    json_root = "/remote-home/pengchengqiu/projects/multilingual-data/data/crawled_data"
    json_paths = []
    for root, dirs, files in os.walk(json_root):
        for file in files:
            if file.endswith('.json'):
                json_paths.append(os.path.join(root, file))
    print("len(json_paths): {}".format(len(json_paths)))
    en_token_num, sp_token_num = 0, 0
    for json_path in json_paths:
        json_file = json.load(open(json_path, 'r'))
        en_token_num += len(json_file['title']["en"])
        for key_name in json_file['content_en']:
             en_token_num += len(json_file['content_en'][key_name]["section_name"])
             en_token_num += len(json_file['content_en'][key_name]["content"])
             
        if "sp" in json_file['title']:
            sp_token_num += len(json_file['title']["sp"])
            sp_token_num += len(json_file['title']["sp"])
            for key_name in json_file['content_sp']:
                sp_token_num += len(json_file['content_sp'][key_name]["section_name"])
                sp_token_num += len(json_file['content_sp'][key_name]["content"])
    print("en_token_num: {}".format(en_token_num//4))
    print("sp_token_num: {}".format(sp_token_num//4))
 
def unzip_file():
    # unzip the zip file
    zip_file_path = '/remote-home/pengchengqiu/data/collections_french.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('../data/')
    print("unzip done!")

if __name__ == "__main__":
    unzip_file()