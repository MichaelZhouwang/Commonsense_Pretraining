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

def run():
    #torch.multiprocessing.freeze_support()
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)
    args_dict = dict(
        data_dir="",  # path for data files
        output_dir="",  # path to save the checkpoints
        model_name_or_path='t5-base',
        checkpoint_dir='',  # checkpoint_dir
        tokenizer_name_or_path='t5-base',
        nsp_generate=False,
        max_seq_length=128,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=4,
        eval_batch_size=4,
        num_train_epochs=2,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args_dict.update({'data_dir': 'commongen', 'output_dir': 'commongen_concept_output_epoch10', 'num_train_epochs': 10})
    args = argparse.Namespace(**args_dict)

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointcheckpoint_ckpt_epoch_*.ckpt"), recursive=True)))
    print(str(checkpoints))

    model = T5FineTuner.load_from_checkpoint(checkpoints[-1])

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    test_examples = [x.rstrip() for x in open('./commongen/test.source').readlines()]
    test_fout = open('test.txt','w')
    val_examples = [x.rstrip() for x in open('./commongen/valid.source').readlines()]
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