import json

test_index = "/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.index.txt"
test_input = "/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.cs_str.txt"
test_tgt = "/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt"

bert_based_out = "/home/bill/CommonGen-plus/methods/BERT-based/bert.test"
gpt2_out = "/home/bill/CommonGen-plus/methods/GPT-2/gpt2.test"
unilm_out = "/home/bill/CommonGen/methods/unilm-based/decoded_sentences_simple/test/model.10.bin.test"
unilmv2_out = "/home/bill/CommonGen-plus/methods/UniLM_v2/unilmv2.test"
bart_out = "/home/bill/CommonGen-plus/methods/BART/bart.test"
t5_out = "/home/bill/CommonGen-plus/methods/T5/t5.test"

brnn_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_brnn.test.src_simple.out"
meanpooling_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_mean.test.src_alpha.out"
leven_out = "/home/bill/CommonGen/methods/fairseq_story/fairseq_local/output/final.leven.alpha.test.txt"
trans_out = "/home/bill/CommonGen/methods/opennmt_based_methods/model_output/commongen_transformer.test.src_simple.out"

reasons_dict = {}
with open("/home/bill/CommonGen-plus/dataset/final_data/commongen.test.jsonl") as f:
    for line in f:
        tmp = json.loads(line)
        reasons_dict[tmp["concept_set"]] = tmp["reason"]

with open(test_index) as index_f, open(test_input) as input_f, open(test_tgt) as tgt_f, \
        open(brnn_out) as brnn_f, open(meanpooling_out) as meanpooling_f, open(leven_out) as leven_f, open(trans_out) as trans_f, \
        open(bert_based_out) as bert_based_f, open(gpt2_out) as gpt2_f, open(unilm_out) as unilm_f, open(unilmv2_out) as unilmv2_f, open(bart_out) as bart_f, open(t5_out) as t5_f, \
        open("all_sentences.txt", "w") as f:

    index_lines = index_f.readlines()
    # print(len(index_lines))
    input_lines = input_f.readlines()
    # print(len(input_lines))
    tgt_lines = tgt_f.readlines()
    # print(len(tgt_lines))

    brnn_lines = brnn_f.readlines()
    # print(len(brnn_lines))
    meanpooling_lines = meanpooling_f.readlines()
    # print(len(meanpooling_lines))
    leven_lines = leven_f.readlines()
    # print(len(leven_lines))
    trans_lines = trans_f.readlines()
    # print(len(trans_lines))

    bert_based_lines = bert_based_f.readlines()
    # print(len(bert_based_lines))
    gpt2_lines = gpt2_f.readlines()
    # print(len(gpt2_lines))
    unilm_lines = unilm_f.readlines()
    # print(len(unilm_lines))
    unilmv2_lines = unilmv2_f.readlines()
    # print(len(unilmv2_lines))
    bart_lines = bart_f.readlines()
    # print(len(bart_lines))
    t5_lines = t5_f.readlines()
    # print(len(t5_lines))

    out = []
    tmp_idx = -1
    previous_string = ""
    tmp_concept = ""
    for list_idx, pack in enumerate(zip(index_lines, input_lines, tgt_lines)):

        if int(pack[0].strip()) != tmp_idx:

            previous_string += "=======================================================================\n"
            f.write(previous_string)
            previous_string = ""

            bert_based_sent = bert_based_lines[list_idx].strip()
            gpt2_sent = gpt2_lines[list_idx].strip()
            unilm_sent = unilm_lines[list_idx].strip()
            unilmv2_sent = unilmv2_lines[list_idx].strip()
            bart_sent = bart_lines[list_idx].strip()
            t5_sent = t5_lines[list_idx].strip()

            brnn_sent = brnn_lines[list_idx].strip()
            meanpooling_sent = meanpooling_lines[list_idx].strip()
            trans_sent = trans_lines[list_idx].strip()
            leven_sent = leven_lines[list_idx].strip()

            human_sent = pack[2].strip()

            tmp_concept = pack[1].strip()
            concepts = "  ".join(pack[1].strip().split("#"))
            reason = reasons_dict[tmp_concept].pop(0)

            previous_string += concepts + "\n" + "\n" \
                          "bert_based: " + bert_based_sent + "\n" + \
                          "gpt2: " + gpt2_sent + "\n" + \
                          "unilm: " + unilm_sent + "\n" + \
                          "unilmv2: " + unilmv2_sent + "\n" + \
                          "bart: " + bart_sent + "\n" + \
                          "t5: " + t5_sent + "\n" + "\n" + \
                          "brnn: " + brnn_sent + "\n" + \
                          "meanpooling: " + meanpooling_sent + "\n" + \
                          "leven: " + leven_sent + "\n" + \
                          "trans: " + trans_sent + "\n" + "\n" + \
                          "ground: " + human_sent + "\n" + \
                               "reason: " + reason + "\n"

            tmp_idx = int(pack[0].strip())
        else:
            tmp_idx = int(pack[0].strip())

            human_sent = pack[2].strip()

            reason = reasons_dict[tmp_concept].pop(0)
            previous_string += "ground: " + human_sent + "\n" + \
                               "reason: " + reason + "\n"





