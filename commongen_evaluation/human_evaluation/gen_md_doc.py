import json
import random
random.seed(42)

with open("human_eval.jsonl", "r") as f:
    lines = f.read().split("\n")

for index, line in enumerate(lines):
    instance = json.loads(line)
    concept_set = instance["input"]
    generations = instance["sentences"]
    random.shuffle(generations)
    
    print("**ID:[%d]** | "%index, "_**Concept Set**_=", ", ".join(concept_set) + "\n")
    # print("| _**Sentence**_   |      GD      |  CP |")
    # print("|"+"-"*100+"|:------:|:------:|")
    for sid, sent in enumerate(generations):
        print( "[%d]: "%sid + "_%s_"%sent[1].lower() + " |  [&nbsp;]   \n\n")
        # print("\n")
    print("\n\n\n")
    instance["references"][:3] = sorted(instance["references"][:3], key=lambda x:len(x)) 
    print("_***References***_\n\n")
    print("- _**%s**_"%instance["references"][0] , "\n")
    print("- _**%s**_"%instance["references"][-1] , "\n")
    print("-"*50 + "\n\n")
    print("\n\n")
    print("x    ")
    print("\n\n")
    # print("-------------------------------------")