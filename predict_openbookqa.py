import torch
import csv
import argparse
from trainer import *
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from dataset_baselines import OBQAProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run():
    #torch.multiprocessing.freeze_support()
    set_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="datasets/openbookqa",
                        help='Path for Data files')
    parser.add_argument('--use_KB', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to use Knowledge Base while constructing the dataset (question/answer) pairs?')
    parser.add_argument('--output_dir', type=str, default="outputs/openbookqa_prediction_outputs",
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="outputs/openbookqa_outputs",
                        help='Checkpoint directory')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-base",
                        help='Tokenizer name or Path')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum Sequence Length')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for Evaluation')

    args = parser.parse_args()
    print(args)

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Creating output directory")

    checkpoints = list(sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch=*.ckpt"), recursive=True)))
    print("Using checkpoint = ", str(checkpoints[-1]))

    t5model = T5FineTuner.load_from_checkpoint(checkpoints[-1])
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    test_csvfile = open(os.path.join(args.output_dir, 'dev.csv'),'w')
    test_writer = csv.writer(test_csvfile)
    proc = OBQAProcessor(args.use_KB)
    test_examples = proc.get_dev_examples(args.data_dir)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    t5model.to(device)

    for batch in tqdm(list(chunks(test_examples, args.eval_batch_size))):
        batch_question = [b.question for b in batch]
        options = [['%s: %s' % (i, option) for i, option in zip('1234', b.answers)] for b in batch]
        options = [" ".join(opts) for opts in options]

        if args.use_KB:
            articles = [b.article for b in batch]
        else:
            articles = None

        inputs = []
        if args.use_KB:
            for article, question, option in zip(articles, batch_question, options):
                inputs.append("article: %s  context: %s  options: %s  </s>" % (article[0:300], question, option))
        else:
            for question, option in zip(batch_question, options):
                inputs.append("context: %s  options: %s </s>" % (question, option))

        dct = tokenizer.batch_encode_plus(inputs, max_length=args.max_seq_length, return_tensors="pt", pad_to_max_length=True, truncation=True)
        outs = t5model.model.generate(input_ids=dct['input_ids'].cuda(),
                                    attention_mask=dct['attention_mask'].cuda(),
                                    max_length=2)

        LABELS = ['A', 'B', 'C', 'D']
        dec = [LABELS[int(tokenizer.decode(ids))-1] for ids in outs]

        for d in dec:
            test_writer.writerow([d])

if __name__ == '__main__':
    run()