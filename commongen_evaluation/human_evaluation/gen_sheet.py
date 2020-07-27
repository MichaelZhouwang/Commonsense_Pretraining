import json
import random
from itertools import combinations
random.seed(42)

with open("human_eval.jsonl", "r") as f:
    lines = f.read().split("\n")

for index, line in enumerate(lines):
    instance = json.loads(line)
    concept_set = instance["input"]
    generations = instance["sentences"]
    random.shuffle(generations)
    instance["references"][:3] = sorted(instance["references"][:3], key=lambda x:len(x)) 
    print("Concept set:", " ".join(concept_set))
    print("Reference:", instance["references"][-1] )
    for s1, s2 in combinations(generations, 2):
        # print("%s\t%s" %(s1[0], s2[0]))
        print("%s\t%s" %(s1[1].lower(), s2[1].lower()))
    print()