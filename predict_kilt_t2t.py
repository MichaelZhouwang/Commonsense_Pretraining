from logger import LoggingCallback
import torch
import argparse
import pytorch_lightning as pl
from dataset import KILTT2TProcessor
from trainer import *
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import random
import numpy as np
import glob
import os
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extractValLoss(checkpoint_path):
    """Eg checkpoint path format: path_to_dir/checkpoint_epoch=4-val_loss=0.450662.ckpt"""

    val_loss = float(re.search('val_loss=(.+?).ckpt', checkpoint_path).group(1))
    return val_loss

def extractStepOREpochNum(checkpoint_path):
    """Eg checkpoint path format: path_to_dir/checkpoint_epoch=4.ckpt (or)
        path_to_dir/checkpoint_epoch=4-step=50.ckpt (or)
    """

    if "step" in checkpoint_path:
        num = int(re.search('step=(.+?).ckpt', checkpoint_path).group(1))
    else:
        num = int(re.search('epoch=(.+?).ckpt', checkpoint_path).group(1))
    return num

def getBestModelCheckpointPath(checkpoint_dir):
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.ckpt"))

    try:
        # Get the checkpoint with lowest validation loss
        sorted_list = sorted(checkpoint_list, key=lambda x: extractValLoss(x.split("/")[-1]))
    except:
        # If validation loss is not present, get the checkpoint with highest step number or epoch number.
        sorted_list = sorted(checkpoint_list, key=lambda x: extractStepOREpochNum(x.split("/")[-1]), reverse=True)

    return sorted_list[0]

def run():
    #torch.multiprocessing.freeze_support()
    set_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="", required=True,
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="", required=True,
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="", required=True,
                        help='Checkpoint directory')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-base",
                        help='Tokenizer name or Path')
    parser.add_argument('--max_source_length', type=int, default=128,
                        help='Maximum Source Length')
    parser.add_argument('--max_target_length', type=int, default=32,
                        help='Maximum Target Length')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for Evaluation')

    args = parser.parse_known_args()[0]
    print(args)

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Creating output directory")

    best_checkpoint_path = getBestModelCheckpointPath(args.checkpoint_dir)
    print("Using checkpoint = ", str(best_checkpoint_path))

    t5model = T5FineTuner.load_from_checkpoint(best_checkpoint_path)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    task_type = args.data_dir.split("/")[-1]
    kilt_proc = KILTT2TProcessor(task_type)
    test_examples = kilt_proc.get_test_examples(args.data_dir)
    test_fout = open(os.path.join(args.output_dir, 'test.txt'),'w')
    val_examples = kilt_proc.get_dev_examples(args.data_dir)
    val_fout = open(os.path.join(args.output_dir, 'val.txt'),'w')

    max_length = args.max_target_length
    min_length = 1

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    t5model.to(device)

    for batch in tqdm(list(chunks(test_examples, args.eval_batch_size))):
        batch_inputs = [b["input"] for b in batch]
        dct = tokenizer.batch_encode_plus(batch_inputs, max_length=args.max_source_length, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = t5model.model.generate(
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

    for batch in tqdm(list(chunks(val_examples, args.eval_batch_size))):
        dct = tokenizer.batch_encode_plus(batch, max_length=args.max_source_length, return_tensors="pt", pad_to_max_length=True, truncation=True)
        summaries = t5model.model.generate(
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