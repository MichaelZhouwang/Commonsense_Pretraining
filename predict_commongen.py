from logger import LoggingCallback
import random
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
from dataset import NSPDataset
from trainer import *
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

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

    parser.add_argument('--data_dir', type=str, default="datasets/commongen",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="outputs/commongen_concept_output_epoch10",
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="",
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

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointcheckpoint_ckpt_epoch_*.ckpt"), recursive=True)))
    print(str(checkpoints))

    model = T5FineTuner.load_from_checkpoint(checkpoints[-1])

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    test_examples = [x.rstrip() for x in open(os.path.join(args.data_dir, 'test.source')).readlines()]
    test_fout = open('test.txt','w')
    val_examples = [x.rstrip() for x in open(os.path.join(args.data_dir, 'valid.source')).readlines()]
    val_fout = open('val.txt','w')

    max_length = 24
    min_length = 1

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    for batch in tqdm(list(chunks(test_examples, 8))):
        dct = tokenizer.batch_encode_plus(batch, max_length=64, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = model.model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=5,
            length_penalty=0.6,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        for hypothesis in dec:
            test_fout.write(hypothesis + "\n")
            test_fout.flush()

    for batch in tqdm(list(chunks(val_examples, 8))):
        dct = tokenizer.batch_encode_plus(batch, max_length=64, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = model.model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=5,
            length_penalty=0.6,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        for hypothesis in dec:
            val_fout.write(hypothesis + "\n")
            val_fout.flush()


if __name__ == '__main__':
    run()