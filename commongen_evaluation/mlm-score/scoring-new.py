from mlm.scorers import MLMScorer, MLMScorerPT,LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import argparse
from tqdm import tqdm 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred', default="", type=str)
args = parser.parse_args()

ctxs = [mx.gpu(6)]



def score_lines(scorer,lines,generate_scores_file):
    scores = []
    sline = lines[0]
    slines = [sline.strip()]
    i = 0
    bsz = 128
    count = 1
    for sline in tqdm(lines[1:]):
        if count % bsz == 0:
            #print(slines)
            res = scorer.score_sentences(slines)
            scores.extend(res)
                # fout.write(hypothesis + '\n')
                # fout.flush()
            slines = []
            i += 1
            # print(i)
        slines.append(sline.strip())
        count += 1
    if slines != []:
        res = scorer.score_sentences(slines)
        scores.extend(res)

    if generate_scores_file:
        with open('scores_'+args.pred,'w',encoding='utf8') as writer:
            for score in scores:
                writer.write(str(score)+'\n')
    assert len(lines) == len(scores)
    return scores


if __name__=='__main__':

    args = parser.parse_args()
    ctxs = [mx.gpu(6)]

    generate_scores_file = False

    models = ['bert-base-en-uncased','roberta-base-en-cased', 'gpt2-345m-en-cased']

    res_lines = open(args.pred,'r',encoding='utf8').readlines()

    for model_name in models:
        print('Evaluating with %s'%model_name)
        model, vocab, tokenizer = get_pretrained(ctxs, model_name)
        if model_name in ['bert-base-en-uncased','roberta-base-en-cased']:
            scorer = MLMScorer(model, vocab, tokenizer, ctxs)
        else:
            scorer = LMScorer(model, vocab, tokenizer, ctxs)

        scores = score_lines(scorer,res_lines,generate_scores_file)

        print("score with %s is %s" %(model_name,str(np.mean(scores))))

