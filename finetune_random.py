from logger import LoggingCallback
from custom_checkpoint import CustomCheckpointCallback
import random
import numpy as np
import torch
import argparse
import os
import pytorch_lightning as pl
from dataset import NSPDataset
from trainer import *

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

    parser.add_argument('--data_dir', type=str, default="datasets/wikitext-2-raw",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="outputs/nsp_new_wiki_random",
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Checkpoint directory')
    parser.add_argument('--save_every_n_steps', type=int, default=10,
                        help='Interval of training steps to save the model checkpoints')

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
    parser.add_argument('--num_train_epochs', type=int, default=2,
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    custom_checkpoint_callback = CustomCheckpointCallback(
        filepath=args.output_dir, prefix="checkpoint_", save_every_n_steps=args.save_every_n_steps
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpu_nums,
        max_epochs=args.num_train_epochs,
        early_stop_callback=args.early_stop_callback,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), custom_checkpoint_callback],
        distributed_backend='ddp'
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


if __name__ == '__main__':
    run()