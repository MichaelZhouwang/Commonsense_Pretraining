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
from dataset import CommonsenseQAProcessor

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

    parser.add_argument('--data_dir', type=str, default="datasets/csqa",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="outputs/csqa",
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="outputs/csqa_output_epoch10",
                        help='Checkpoint directory')

    parser.add_argument('--model_name_or_path', type=str, default="t5-base",
                        help='Model name or Path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-base",
                        help='Tokenizer name or Path')
    parser.add_argument('--nsp_generate', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to generate NSP?')

    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    parser.add_argument('--opt_level', type=str, default="01",
                        help='Optimization level')
    parser.add_argument('--early_stop_callback', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to do early stopping?')

    # if you want to enable 16-bit training then install apex and set this to true
    parser.add_argument('--fp_16', type=lambda x: (str(x).lower() == 'true'), default="True",
                        help='Whether to use 16 bit precision floating point operations?')

    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='Epsilon value for Adam Optimizer')

    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum Gradient Norm value for Clipping')


    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum Sequence Length')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Batch size for Training')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for Evaluation')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Number of Training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Gradient Accumulation Steps')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs to use for computation')
    parser.add_argument('--gpu_nums', type=str, default="0",
                        help='GPU ids separated by "," to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Manual Seed Value')

    args = parser.parse_args()
    print(args)

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Creating output directory")

    checkpoints = list(sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch=*.ckpt"), recursive=True)))
    print("Using checkpoint = ", str(checkpoints[-1]))

    t5model = T5FineTuner.load_from_checkpoint(checkpoints[-1])
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    test_csvfile = open(os.path.join(args.output_dir, 'dev.csv'),'w')
    test_writer = csv.writer(test_csvfile)
    proc = CommonsenseQAProcessor('rand')
    test_examples = proc.get_dev_examples(args.data_dir)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    t5model.to(device)

    for batch in tqdm(list(chunks(test_examples, args.eval_batch_size))):
        batch_question = [b.question for b in batch]
        options = [['%s: %s' % (i, option) for i, option in zip('12345', b.answers)] for b in batch]
        options = [" ".join(opts) for opts in options]

        inputs = []
        for question, option in zip(batch_question, options):
            inputs.append("context: %s  options: %s </s>" % (question, option))

        dct = tokenizer.batch_encode_plus(inputs, max_length=args.max_seq_length, return_tensors="pt", pad_to_max_length=True, truncation=True)
        outs = t5model.model.generate(input_ids=dct['input_ids'].cuda(),
                                    attention_mask=dct['attention_mask'].cuda(),
                                    max_length=2)

        LABELS = ['A', 'B', 'C', 'D', 'E']
        dec = [LABELS[int(tokenizer.decode(ids))-1] for ids in outs]
        ids = [b.qid for b in batch]

        for i, d in zip(ids, dec):
            test_writer.writerow([i,d])



if __name__ == '__main__':
    run()