import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import SummarizationDataset, CSQADataset, PIQADataset, ANLIDataset, OBQADataset, KILTFEVERDataset, KILTT2TDataset
import argparse
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


def get_dataset(tokenizer, type_path, args):
    print(args.data_dir)
    data_dir_leaf = args.data_dir.split("/")[-1]
    # chunshu : 128 / 128
    # dong-ho : 256 / 128
    if data_dir_leaf == 'commongen' or data_dir_leaf == 'commongen_20' or data_dir_leaf == 'commongen_40' or data_dir_leaf == 'commongen_60' or data_dir_leaf == 'commongen_80' or data_dir_leaf == 't5_processed':
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_source_length, max_target_length=args.max_target_length)
    elif data_dir_leaf == "keyword_lm" or data_dir_leaf == "concept_deshuffling":
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,max_source_length=args.max_seq_length, max_target_length=args.max_seq_length)
    elif data_dir_leaf == 'option1': # choice of string
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)
    elif data_dir_leaf == 'option2': # string of choice
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=int(args.max_seq_length / 2))
    elif data_dir_leaf == 'option3': # True / False
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=2)
    elif data_dir_leaf == 'mixed_dataset_key_lm_concept':
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=args.max_seq_length)
    elif data_dir_leaf == 'mixed_dataset_key_lm_concept_option2':
        return SummarizationDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_seq_length, max_target_length=int(args.max_seq_length / 2))

    elif data_dir_leaf == 'csqa' or data_dir_leaf == 'csqa_20' or data_dir_leaf == 'csqa_40' or data_dir_leaf == 'csqa_60' or data_dir_leaf == 'csqa_80':
        return CSQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == 'piqa':
        return PIQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "anli":
        return ANLIDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "openbookqa" or data_dir_leaf == "openbookqa_20" or data_dir_leaf == "openbookqa_40" or data_dir_leaf == "openbookqa_60" or data_dir_leaf == "openbookqa_80":
        return OBQADataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length, use_KB=args.use_KB)

    # KILT Tasks
    elif data_dir_leaf == "kilt_fever":
        return KILTFEVERDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)
    elif data_dir_leaf == "kilt_natural_qa" or data_dir_leaf == "kilt_ay2" or data_dir_leaf == "kilt_trivia_qa":
        return KILTT2TDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_source_length=args.max_source_length, max_target_length=args.max_target_length, createMultipleSamples=args.expandSamples)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        print("Model params: ", self.hparams)

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True #temporary fix (only work at single GPU env)

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=16)

        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        print(len(dataloader.dataset))
        print(t_total)
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=16)
