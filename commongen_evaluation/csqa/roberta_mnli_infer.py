from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from fairseq.data.data_utils import collate_tokens
import torch
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np



# Load data 
questions = open("csqa.dev.questions").read().split("\n")
answers = open("csqa.dev.answers").read().split("\n")
# model_generations = open("csqa.dev.qac.bart.res").read().split("\n")
model_generations = open("csqa.dev.choices").read().split("\n")



# batch_of_pairs = [
#     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
#     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
#     ['potatoes are awesome.', 'I like to run.'],
#     ['Mars is very far from earth.', 'Mars is very close.'],
# ] 

batch_of_pairs = [(q,a) for q,a in zip(questions, model_generations)]

print("loading model")
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()
roberta.cuda()

print("encoding data")
eval_dataset = collate_tokens(
    [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=16)


preds = None

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    logprobs = roberta.predict('mnli', batch)
    if preds is None:
        preds = logprobs.detach().cpu().numpy()
    else:
        preds = np.append(preds, logprobs.detach().cpu().numpy(), axis=0)
 

# print(preds)
# print(preds.argmax(axis=1))
scores = [p[2]-p[0] for p in preds]  
grouped_scores = list(zip(*(iter(scores),) * 5))

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
correct = 0
for i in range(len(answers)):
    truth = int(answers[i])
    pred = argmax(grouped_scores[i])
    if truth == pred:
        correct += 1
print(correct/len(answers))

# [0, 2, 1, 0]
# 0 means conflict 
# 1 means neutral 
# 2 means entailment