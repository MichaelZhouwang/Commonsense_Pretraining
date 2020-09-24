from generator.concept.concept_generator import *
from tqdm import tqdm
import argparse

generator = ConceptGenerator()

def match_sents(sent_path):
    result = []
    #assert sent_path.endswith(".txt")
    with open(sent_path) as f:
        sents = f.read().split("\n")

    for index, sent in enumerate(tqdm(sents)):
        if generator.check_availability(sent):
            generated_sentence = generator.generate(sent)

            result.append({"original_sentence": sent, "output_sentence": generated_sentence})
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='SWAG/swag_sent.txt')
parser.add_argument('--save_path', default="SWAG/swag_sent.matched.json")

args = parser.parse_args()

batch_result = match_sents(sent_path=args.input_path)
result_path = '%s.%d-of-%d' % (args.save_path, args.batch_id, args.num_batch)

with open(result_path, "w") as f:
    for line in batch_result:
        f.write("correct the order of the following sentence : "+line["output_sentence"]+"\t"+line["original_sentence"]+'\n')
