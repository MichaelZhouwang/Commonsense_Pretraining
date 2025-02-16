from logger import LoggingCallback
from custom_checkpoint import CustomCheckpointCallback
import random
import numpy as np
import torch
import argparse
import os
import re
import glob
import pytorch_lightning as pl
from trainer import *

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="", required=True,
                        help='Path for Data files')
    parser.add_argument('--expandSamples', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to create multiple instances for input sequences with multiple correct answers?')
    parser.add_argument('--output_dir', type=str, default="", required=True,
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Checkpoint directory')
    parser.add_argument('--save_every_n_steps', type=int, default=-1,
                        help='Interval of training steps to save the model checkpoints. Use -1 to disable this callback')

    parser.add_argument('--model_name_or_path', type=str, default="t5-base",
                        help='Model name or Path')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-base",
                        help='Tokenizer name or Path')

    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    parser.add_argument('--opt_level', type=str, default="O1",
                        help='Optimization level')
    parser.add_argument('--early_stop_callback', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to do early stopping?')

    # if you want to enable 16-bit training then install apex and set this to true
    parser.add_argument('--fp_16', type=lambda x: (str(x).lower() == 'true'), default="False",
                        help='Whether to use 16 bit precision floating point operations?')

    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='Epsilon value for Adam Optimizer')

    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum Gradient Norm value for Clipping')

    parser.add_argument('--max_source_length', type=int, default=128,
                        help='Maximum Source Length')
    parser.add_argument('--max_target_length', type=int, default=32,
                        help='Maximum Target Length')
    parser.add_argument('--warmup_steps', type=int, default=400,
                        help='Number of warmup steps')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Batch size for Training')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for Evaluation')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Number of Training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient Accumulation Steps')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs to use for computation')
    parser.add_argument('--gpu_nums', type=str, default="0",
                        help='GPU ids separated by "," to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Manual Seed Value')

    args = parser.parse_known_args()[0]
    print(args)

    set_seed(args.seed)

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Creating output directory")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir + "/{epoch}-{val_loss:.6f}", prefix="checkpoint_", monitor="val_loss", mode="min", save_top_k=1
    )

    trainer_custom_callbacks = [LoggingCallback()]
    if args.save_every_n_steps != -1:
        custom_checkpoint_callback = CustomCheckpointCallback(
            filepath=args.output_dir, prefix="checkpoint_", save_every_n_steps=args.save_every_n_steps
        )
        trainer_custom_callbacks.append(custom_checkpoint_callback)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpu_nums,
        max_epochs=args.num_train_epochs,
        early_stop_callback=args.early_stop_callback,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=trainer_custom_callbacks,
        distributed_backend='ddp'
    )

    if len(args.checkpoint_dir) != 0:
        best_checkpoint_path = getBestModelCheckpointPath(args.checkpoint_dir)
        print("Using checkpoint = ", str(best_checkpoint_path))
        checkpoint_state = torch.load(best_checkpoint_path, map_location="cpu")
        model = T5FineTuner(args)
        model.load_state_dict(checkpoint_state['state_dict'])
    else:
        model = T5FineTuner(args)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


if __name__ == '__main__':
    run()