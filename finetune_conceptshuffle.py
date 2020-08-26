from logger import LoggingCallback
import random
import numpy as np
import torch
import argparse
import pytorch_lightning as pl
from dataset import NSPDataset
from trainer import *

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
        concept_generate = True,
        max_seq_length=128,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=4,
        eval_batch_size=4,
        num_train_epochs=2,
        gradient_accumulation_steps=16,
        n_gpu=3,
        early_stop_callback=False,
        fp_16=True,  # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args_dict.update({'data_dir': 'wikitext-2-raw', 'output_dir': 'nsp_new_wiki_generate', 'num_train_epochs': 2})
    args = argparse.Namespace(**args_dict)
    print(args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus="1,2,3",
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
        distributed_backend='ddp'
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


if __name__ == '__main__':
    run()