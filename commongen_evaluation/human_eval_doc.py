import json

test_index = "~/CommonGen-plus/dataset/final_data/commongen/commongen.test.index.txt"
test_input = "~/CommonGen-plus/dataset/final_data/commongen/commongen.test.cs_str.txt"
test_tgt = "~/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt"
test_data_path = "~/CommonGen-plus/dataset/final_data/commongen.test.jsonl"

model_paths = {}

model_paths["BERT-Gen"] = "~/CommonGen-plus/methods/BERT-based/bert.test"
model_paths["GPT-2"] = "~/CommonGen-plus/methods/GPT-2/gpt2.test"
model_paths["UniLM"] = "~/CommonGen-plus/methods/unilm_based/decoded_sentences/test/model.10.bin.test" 
model_paths["BART"] = "~/CommonGen-plus/methods/BART/fairseq_local/bart.test"
model_paths["T5"] = "~/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test"

# brnn_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_brnn.test.src_simple.out"
# meanpooling_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_mean.test.src_alpha.out"
# leven_out = "/home/bill/CommonGen/methods/fairseq_story/fairseq_local/output/final.leven.alpha.test.txt"
# trans_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_transformer.test.src_simple.out"

reasons_dict = {}
with open(test_data_path) as f:
    for line in f:
        tmp = json.loads(line)
        reasons_dict[tmp["concept_set"]] = tmp["reason"]

