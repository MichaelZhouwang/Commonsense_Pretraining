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
  


def get_sum_all_dict(annotation_path):
    sum_all_dict = defaultdict(lambda: [0]*100)
    with open(annotation_path) as f:
        index = 0
        group_count = 0
        group = False
        comparsions_better_than_me = defaultdict(list)
        idx = 0
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
                        #print(idx)
                        sum_all_dict[model][idx] = rank
                    idx += 1
                    comparsions_better_than_me = defaultdict(list)
    return sum_all_dict



def compute_w(data):
    """ Computes kendall's W from a list of rating lists.
    0 indicates no agreement and 1 indicates unanimous agreement.
    Parameters
    ---------
    data : list
        List of lists with shape (n_items * n_annotators)
    Return
    ---------
    W : float
        Kendall's W [0:1]
    Example
    ---------
    annotations = [
        [1, 1, 1, 2], # item 1
        [2, 2, 2, 3], # item 2
        [3, 3, 3, 1], # item 3
    ]
    # Annotator #4 disagrees with the other annotators
    # Annotators #1, #2, #3 agree
    W = kendall_w(annotations)
    # output: 0.4375
    """

    assert isinstance(data, list), "You must pass a python list,\
        {} found".format(type(data))
    assert all(isinstance(x, list) for x in data), "You must pass a list of\
        python lists as input."  # To test
    assert all(isinstance(x[y], int) for x in data for y in range(len(x))), "You must\
        pass a list of lists of integers."  # To test

    # Number of annotators
    m = len(data[0])
    # Tests
    if not all(len(i) == m for i in data):
        raise ValueError("Items must all have the same number of annotators.\
            At least one sublist of argument 'data' has different length than\
            the first sublist.")
    if m <= 1:
        raise ValueError("Kendall's W is irrevelent for only one annotator,\
            try adding more lists to argument 'data'.")
    if m == 2:
        warnings.warn("Kendall's W is adapted to measure agreement between\
            more than two annotators. The results might not be reliable in\
            this case.", Warning)

    # Number of items
    n = len(data)
    # Tests
    if n <= 1:
        raise ValueError("Kendall's W is irrevelent for only one item,\
            try adding more sublists to argument 'data'.")

    # Sum of each item ranks
    sums = [sum(x) for x in data]
    # Mean of ranking sums
    Rbar = sum(sums) / n
    # Sum of squared deviations from the mean
    S = sum([(sums[x] - Rbar) ** 2 for x in range(n)])

    W = (12 * S) / (m ** 2 * (n ** 3 - n))

    return W







sum_all_1 = get_sum_all_dict("human_eval_1.tsv")
sum_all_2 = get_sum_all_dict("human_eval_2.tsv")
sum_all_3 = get_sum_all_dict("human_eval_3.tsv")
sum_all_4 = get_sum_all_dict("human_eval_4.tsv")
sum_all_5 = get_sum_all_dict("human_eval_5.tsv")

assert sum_all_1.keys() == sum_all_2.keys() == sum_all_3.keys() == sum_all_4.keys() 
assert len(sum_all_1[0]) == len(sum_all_2[0]) == len(sum_all_3[0]) == len(sum_all_4[0])
num_instances = len(sum_all_1[0])

annotations = []
for i in range(num_instances):
    for model in sum_all_1:
        rank_a, rank_b, rank_c, rank_d, rank_e = sum_all_1[model][i],sum_all_2[model][i], sum_all_3[model][i], sum_all_4[model][i], sum_all_5[model][i]
        annotations.append([rank_a,rank_b,rank_c,rank_d,rank_e])
        
W = []
for i in range(num_instances):
    W.append(compute_w(annotations[i*6:(i+1)*6]))
    
print('Kendall\'s W is %s'%str(np.mean(W)))