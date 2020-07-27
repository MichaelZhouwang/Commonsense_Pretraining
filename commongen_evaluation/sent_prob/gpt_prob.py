import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import argparse
from collections import defaultdict
from collections import Counter
import spacy
import json

nlp = spacy.load('en_core_web_sm')
nlp.pipeline = [('tagger', nlp.tagger)]

def model_init(model_string, cuda):
    if model_string.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    model.eval()
    if cuda:
        model.to('cuda')
    print("Model init")
    return model, tokenizer


def all_cover(concept_set, sentence):
    lemmas = set([t.lemma_ for t in nlp(sentence)])
    bool_list = [c in lemmas for c in concept_set]
    if sum(bool_list) == len(concept_set):
        return True
    return False

def sent_scoring(model_tokenizer, text, cuda):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    assert model is not None
    assert tokenizer is not None
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob


if __name__ == '__main__':
    cuda = True
    # gpt_model, tokenizer = model_init('gpt2', cuda)
 
    model_paths = {}
    model_paths["bRNN"] = "/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_brnn.test.src_alpha.out"
    model_paths["MeanPool"] = "/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_mean.test.src_alpha.out"
    model_paths["Leven-Const"] = "/home/bill/CommonGen-plus/methods/const-levt/const-levt.test"
    model_paths["BERT-Gen"] = "/home/bill/CommonGen-plus/methods/BERT-based/bert.test"
    model_paths["GPT-2"] = "/home/bill/CommonGen-plus/methods/GPT-2/gpt2.test"
    model_paths["UniLM"] = "/home/bill/CommonGen-plus/methods/unilm_based/decoded_sentences/test/model.10.bin.test" 
    model_paths["BART"] = "/home/bill/CommonGen-plus/methods/BART/fairseq_local/bart.test"
    model_paths["T5"] = "/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test"

    index_dict = open("/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.index.txt").read().split()
    print(len(index_dict))

    instances = defaultdict(list)
    human_refs = defaultdict(list)

    for model_name, file_name in model_paths.items():
        with open(file_name) as f:
            check = set()
            for index, line in enumerate(f.readlines()):
                if index_dict[index] not in check:
                    instances[index_dict[index]].append((model_name, line.strip()))
                    check.add(index_dict[index])

    with open("/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt") as f:
        for index, line in enumerate(f.readlines()):
            human_refs[index_dict[index]].append(line.strip())

    input_concepts = defaultdict(list)
    with open("/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt") as f:
        for index, line in enumerate(f.readlines()):
            input_concepts[index_dict[index]] = line.split()

    top_model_count = defaultdict(lambda: 0)
    kept_keys = []
    output_doc = []
    for key, item in tqdm(instances.items(), total=len(instances)):
        # print(key)

        # make sure all sentences are 
        concept_set = input_concepts[key]
        # covers = [all_cover(concept_set, s[1]) for s in item]
        # if not all(covers):
        #     continue  
        
        # mini_model = ""
        # mini_score = None
        ranks = []
        for model, pred in item:
            # loss_score = sent_scoring((gpt_model, tokenizer), pred, cuda)
            loss_score = 0.0
            ranks.append((model, loss_score))
        ranks.sort(key=lambda x: x[1])

        # mean_human_scores = sum([sent_scoring((gpt_model, tokenizer), ref, cuda) for ref in human_refs[key]])/len(human_refs[key])
        if True: # mean_human_scores < ranks[0][1] or 
            kept_keys.append(key)
            output_doc.append({"data_index":key, "input": concept_set, "sentences": item, "references":human_refs[key]})
            for mini_model, _ in ranks[:3]:
                top_model_count[mini_model] += 1
    print(len(kept_keys))
    print(top_model_count)

    with open("qua_eval.jsonl", "w") as f:
        f.write("\n".join([json.dumps(item) for item in output_doc]))
    

    # pred_ppl = 0.0
    # truth_ppl = 0.0
    # cnt_smaller = 0
    # cnt_all = 0
    # r_ppl_dict = {}
    # for (m, p), r in tqdm(zip(pred, ref), total=len(pred)):
    #     p_ppl = sent_scoring((model, tokenizer), p, cuda)
    #     if r not in r_ppl_dict:
    #         r_ppl = sent_scoring((model, tokenizer), r, cuda)
    #         r_ppl_dict[r] = r_ppl
    #     else:
    #         r_ppl = r_ppl_dict[r]
    #     pred_ppl += p_ppl
    #     truth_ppl += r_ppl
    #     if p_ppl < r_ppl:
    #         cnt_smaller += 1
    #     cnt_all += 1
    #     print(pred_ppl, truth_ppl, cnt_smaller/cnt_all)

    # print("Final:", pred_ppl, truth_ppl)
    # print("# examples that pred has lower ppl", cnt_smaller)

# CUDA_VISIBLE_DEVICES=0 python gpt_prob.py  --pred ../../methods/T5/t5.test   --ref ../../dataset/final_data/commongen/commongen.test.tgt.txt
# 60
# CUDA_VISIBLE_DEVICES=1 python gpt_prob.py  --pred ../../methods/UniLM_v2/unilmv2.test   --ref ../../dataset/final_data/commongen/commongen.test.tgt.txt
# 35
# CUDA_VISIBLE_DEVICES=2 python gpt_prob.py  --pred ../../methods/GPT-2/gpt2.test   --ref ../../dataset/final_data/commongen/commongen.test.tgt.txt
# 50
# CUDA_VISIBLE_DEVICES=3 python gpt_prob.py  --pred ../../methods/BART/bart.test   --ref ../../dataset/final_data/commongen/commongen.test.tgt.txt
# 70