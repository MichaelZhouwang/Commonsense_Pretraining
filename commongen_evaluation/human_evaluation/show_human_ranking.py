import numpy as np
from collections import defaultdict

model_index = {"Leven-Const": 0, "GPT-2": 1, "BERT-Gen": 2, "UniLM": 3, "BART": 4, "T5": 5}
inversed_model_index = dict(map(reversed, model_index.items()))
pairs_list = []

with open("model_pairs.tsv") as f:
    for line in f.readlines():
        line = line.strip()
        if line != "":
            pairs_list.append((model_index[line.split("\t")[0]], model_index[line.split("\t")[1]]))


def show_results(anntation_path):
    output_res = []
    index = 0
    sum_all = defaultdict(lambda: [0]*6)
    with open(anntation_path) as f:
        group_count = 0
        group = False
        comparsions_better_than_me = defaultdict(list)
        for line in f.readlines():
            line = line.strip()
            if group:
                if group_count == 15:
                    group_count = 0
                    group = False
                    continue

                model_pair = pairs_list[index]
                data = line.split("\t")
                s1 = data[0]
                s2 = data[1]
                score = float(data[2])
                
                flag = False
                if score > 0.5:
                    better, worse = model_pair[0], model_pair[1] 
                    flag = True
                elif score < 0.5:
                    worse, better = model_pair[0], model_pair[1]
                    flag = True
                if flag:
                    comparsions_better_than_me[inversed_model_index[worse]].append(inversed_model_index[better])
                index += 1
                group_count += 1
            else:
                if line.startswith("Concept set:"):
                    concept_set = line.strip().replace("Concept set:", "").split()
                if line.startswith("Reference:"):
                    group = True
                    # print(comparsions_better_than_me)
                    for model in model_index:
                        rank = len(comparsions_better_than_me.get(model, [])) 
                        sum_all[model][rank] += 1
                    comparsions_better_than_me = defaultdict(list)

    for model in sum_all:
        # print(model)
        cur_sum = 0
        results = []
        for index, k in enumerate(sum_all[model]):
            cur_sum += k 
            # print("top %d"%(index+1), cur_sum/100)
            results.append(cur_sum/100)
        print(",".join([model]+[str(i) for i in results]))


show_results("human_eval_1.tsv")
print()
show_results("human_eval_2.tsv")
print()
show_results("human_eval_3.tsv")
print()
show_results("human_eval_4.tsv")
print()
show_results("human_eval_5.tsv")
    
# print(model_compare)
# print(model_chosen)
# print(model_chosen/model_compare)
# results = model_chosen/model_compare 
# print(",".join([inversed_model_index[i] for i in range(6)]))
# for i in range(len(model_index)):
#     print(",".join([str(j) for j in results[i]]))
