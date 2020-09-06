import os
import pytorch_lightning as pl

class CustomCheckpointCallback(pl.Callback):
    """
        Saves the model checkpoints once every n training steps. Currently saving works only if
        the output directory is empty (so make sure to empty the output directory or choose a new output directory)
    """
    def __init__(self, filepath, prefix, save_every_n_steps):
        self.filepath = filepath
        self.prefix = prefix
        self.save_every_n_steps = save_every_n_steps

    def on_batch_end(self, trainer, pl_module):
        cur_epoch = trainer.current_epoch
        cur_global_step = trainer.global_step
        if cur_global_step % self.save_every_n_steps == 0:
            file_name = self.prefix + "epoch=" + str(cur_epoch) + "_step=" + str(cur_global_step) + ".ckpt"
            ckpt_path = os.path.join(self.filepath, file_name)
            if not os.path.exists(ckpt_path):
                print("Saving model checkpoint at ", ckpt_path)
                trainer.save_checkpoint(ckpt_path)