"""
@Description :   Rationale similarity calculation, including BLEU, BertScore, Rouge
                 Note that rationale similarity is calculated on the cases with Human Checked Passed Rationale.
@Author      :   Henrychur 
@Time        :   2024/09/13 10:17:34
"""
import pandas as pd
import sys
sys.setrecursionlimit(10000) 

METRIC = 'BLEU' # [BLEU, BertScore, Rouge]

if METRIC == 'BLEU':
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import jieba
    from janome.tokenizer import Tokenizer
elif METRIC == 'BertScore':
    from bert_score import score
elif METRIC == 'Rouge':
    from rouge import Rouge 
    import jieba
    from janome.tokenizer import Tokenizer

filepaths = [
    'res_llama_finetune+\\USMLE_en_res.txt',
    'res_llama_finetune+\\USMLE_zh_res.txt',
    'res_llama_finetune+\\lgakuQA_res.txt',
    'res_llama_finetune+\\FrenchMedMCQA_res.txt',
    'res_llama_finetune+\\RuMedDaNet_res.txt',
    'res_llama_finetune+\\Head-QA_res.txt',
]

def get_lang(file_path):
    if file_path.split("\\")[-1] == 'USMLE_en_res.txt':
        return "en"
    elif file_path.split("\\")[-1] == 'USMLE_zh_res.txt' :
        return "zh"
    elif file_path.split("\\")[-1] == 'lgakuQA_res.txt':
        return "jp"
    elif file_path.split("\\")[-1]  == 'FrenchMedMCQA_res.txt':
        return "fra"
    elif file_path.split("\\")[-1] == 'RuMedDaNet_res.txt':
        return "ru"
    elif file_path.split("\\")[-1] == 'Head-QA_res.txt':
        return "spa"


def cal_similarity(line, lang='en'):
    response, gt_answer, gt_rationale = line.split("[SPLIT]")[0], line.split("[SPLIT]")[1], line.split("[SPLIT]")[2]    
    response_rationale = response.split("###Answer: OPTION ")[0].replace("###Rationale: ", "").strip()
    if METRIC == 'BLEU':
        if lang in ['en', 'spa', 'fra', 'ru']:
            reference_tokenized = word_tokenize(gt_rationale)
            candidate_tokenized = word_tokenize(response_rationale)
        elif lang in ['zh']:
            reference_tokenized = list(jieba.cut(gt_rationale))
            candidate_tokenized = list(jieba.cut(response_rationale))
        elif lang in ['jp']:
            t = Tokenizer()
            reference_tokenized = [token.surface for token in t.tokenize(gt_rationale)]
            candidate_tokenized = [token.surface for token in t.tokenize(response_rationale)]

        # 计算BLEU分数
        smoothie = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokenized], candidate_tokenized, smoothing_function=smoothie)
        return bleu_score
    
    elif METRIC == 'Rouge':
        if lang == 'zh':
            gt_rationale = " ".join(jieba.cut(gt_rationale))
            response_rationale = " ".join(jieba.cut(response_rationale))
        elif lang == 'jp':
            t = Tokenizer()
            gt_rationale = " ".join([token.surface for token in t.tokenize(gt_rationale)])
            response_rationale = " ".join([token.surface for token in t.tokenize(response_rationale)])
            # print(gt_rationale )
        rouge = Rouge()
        rouge_scores = rouge.get_scores(response_rationale, gt_rationale)
        
        # [rouge-l, rouge-1, rouge-2]
        return rouge_scores[0]['rouge-l']['f']
    
    elif METRIC == 'BertScore':
        if lang == 'jp':
            lang == 'ja'
        elif lang == 'spa':
            lang == 'es'
        elif lang == 'fra':
            lang == 'fr'
        P, R, F1 = score([response_rationale], [gt_rationale], lang=lang, model_type="bert-base-multilingual-cased")
        f1_score = F1.item()
        return f1_score 
    else:
        raise ValueError("Invalid METRIC")

    
def main():
    scores = []
    # Note that ratinale similarity is calculated on the cases with Human Checked Rationale!
    for filepath in filepaths:
        lang = get_lang(filepath)
        cnt, total_similarity = 0, 0.
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                item_similarity = cal_similarity(line, lang)
                if item_similarity != -1:
                    total_similarity += item_similarity
                    cnt += 1
        scores.append(total_similarity/cnt)
        print("{}\t{:.3f}\t{}\t{:.3f}".format(filepath, total_similarity, cnt, total_similarity/cnt))
    print("AVE:\t", sum(scores)/len(scores))
if __name__ == '__main__':
    main()
    
    # gt_rationale = 'C命题是正确的，根据比尔-朗伯定律，只要溶液足够稀释以避免猝灭效应，发射的荧光强度与荧光物质的浓度成正比。这样可以精确测定分析物的浓度。D命题也是正确的。要使分子呈现荧光现象，它们必须在激发后迅速返回到低能态（基态），通常在纳秒至几十纳秒的范围内。如果激发态寿命过长，非辐射过程如内转换或交叉系统将趋于主导，这将导致磷光而不是荧光。'
    # response_rationale = '荧光测量法是一种利用荧光来测量物质浓度的分析技术。选项A是正确的，因为激发波长被选择为小于待分析分子的发射波长，这有助于保持激发态处于单线态。选项C也是正确的，因为荧光强度与荧光物质浓度成正比，这对于定量分析是必要的。选项E是正确的，因为大多数分子都可能具有荧光性，这使得荧光测量法对于定量和特征化物质非常有用。选项B和D是不正确的：选项B是不正确的，因为荧光现象涉及单线态电子，选项D是不正确的，因为激发态寿命可能非常短，通常小于十纳秒，这是荧光发生所必需的。 答案：选项A，C，E是正确的。'
    # reference_tokenized = list(jieba.cut(gt_rationale))
    # candidate_tokenized = list(jieba.cut(response_rationale))
    # # 计算BLEU分数
    # smoothie = SmoothingFunction().method1
    # weights = (1, 0, 0, 0)
    # bleu_score = sentence_bleu([reference_tokenized], candidate_tokenized, smoothing_function=smoothie, weights=weights)
    # print(bleu_score)