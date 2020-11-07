import os
import pytorch_lightning as pl

class CustomCheckpointCallback(pl.Callback):
    """
        Saves the model checkpoints once every n training steps.
    """
    def __init__(self, filepath, prefix, save_every_n_steps):
        self.filepath = filepath
        self.prefix = prefix
        self.save_every_n_steps = save_every_n_steps
        self.ckpt_hash_paths = dict()

    def on_batch_end(self, trainer, pl_module):
        cur_epoch = trainer.current_epoch
        cur_global_step = trainer.global_step
        if cur_global_step % self.save_every_n_steps == 0:
            file_name = self.prefix + "epoch=" + str(cur_epoch) + "-step=" + str(cur_global_step) + ".ckpt"
            ckpt_path = os.path.join(self.filepath, file_name)

            # Store the trainer only once for every ckpt path name:
            if self.ckpt_hash_paths.get(ckpt_path) is None:
                self.ckpt_hash_paths[ckpt_path] = True
                print("Saving model checkpoint at ", ckpt_path)
                trainer.save_checkpoint(ckpt_path)


class EpochEndCheckpointCallback(pl.Callback):
    """
        Saves the model checkpoints once every n epochs.
    """
    def __init__(self, filepath, prefix, save_every_n_epochs):
        self.filepath = filepath
        self.prefix = prefix
        self.save_every_n_epochs = save_every_n_epochs
        self.ckpt_hash_paths = dict()

    def on_epoch_end(self, trainer, pl_module):
        cur_epoch = trainer.current_epoch
        cur_global_step = trainer.global_step
        if cur_epoch % self.save_every_n_epochs == 0:
            file_name = self.prefix + "epoch=" + str(cur_epoch) + ".ckpt"
            ckpt_path = os.path.join(self.filepath, file_name)

            # Store the trainer only once for every ckpt path name:
            if self.ckpt_hash_paths.get(ckpt_path) is None:
                self.ckpt_hash_paths[ckpt_path] = True
                print("Saving model checkpoint at ", ckpt_path)
                trainer.save_checkpoint(ckpt_path)