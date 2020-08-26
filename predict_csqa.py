import torch
import csv

from trainer import *
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from dataset import CommonsenseQAProcessor

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

    args_dict.update({'data_dir': 'csqa', 'output_dir': 'model_save/csqa', 'num_train_epochs': 10})
    args = argparse.Namespace(**args_dict)

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointcheckpoint_ckpt_epoch_*.ckpt"), recursive=True)))
    print(str(checkpoints))

    model = T5FineTuner.load_from_checkpoint(checkpoints[0])
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    test_csvfile = open('dev.csv','w')
    test_writer = csv.writer(test_csvfile)
    proc = CommonsenseQAProcessor('rand')
    test_examples = proc.get_dev_examples('csqa')

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    for batch in tqdm(list(chunks(test_examples, 8))):
        batch_question = [b.question for b in batch]
        options = [['%s: %s' % (i, option) for i, option in zip('12345', b.answers)] for b in batch]
        options = [" ".join(opts) for opts in options]

        inputs = []
        for question, option in zip(batch_question, options):
            inputs.append("context: %s  options: %s </s>" % (question, option))

        dct = tokenizer.batch_encode_plus(inputs, max_length=128, return_tensors="pt", pad_to_max_length=True, truncation=True)
        outs = model.model.generate(input_ids=dct['input_ids'].cuda(),
                                    attention_mask=dct['attention_mask'].cuda(),
                                    max_length=2)

        LABELS = ['A', 'B', 'C', 'D', 'E']
        dec = [LABELS[int(tokenizer.decode(ids))-1] for ids in outs]
        ids = [b.qid for b in batch]

        for i, d in zip(ids, dec):
            test_writer.writerow([i,d])



if __name__ == '__main__':
    run()