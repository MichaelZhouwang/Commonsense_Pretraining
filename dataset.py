# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import uuid
import tensorflow.compat.v1 as tf
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os, glob

import logging
import numpy as np
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

tf.config.experimental.set_visible_devices([], 'GPU')
tf.enable_eager_execution()
import warnings
warnings.filterwarnings("ignore", category=Warning)

class NSPDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, nsp_generate=False, max_len=512):

        self.file_path = os.path.join(data_dir)
        self.files = glob.glob("%s/wiki.%s.raw" % (self.file_path, type_path))

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.nsp_generate = nsp_generate
        if self.nsp_generate:
            model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)
            self.gpt_tokenizer = tokenizer_class.from_pretrained('gpt2')
            self.gpt_model = model_class.from_pretrained('gpt2')
            self.gpt_model.to('cuda')
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        self._buil_examples_from_files(self.files)

    def neighboring_pairs(self, dataset, text_key='text', reuse_sentences=True):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""
            def my_fn(text):
                lines = tf.strings.split([text], sep='\n').values
                return tf.strings.strip(lines)
            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_into_pairs(line):
            """Split a given text example into pairs of neighboring sentences."""
            # TODO(mmatena): Use better sentence segmentation.
            sep = str(uuid.uuid4())
            sentences = tf.strings.regex_replace(line, r'((?:\.|\!|\?)+)', r'\1' + sep)
            sentences = tf.strings.strip(tf.strings.split([sentences], sep).values)
            if reuse_sentences:
                firsts = sentences[:-1]
                seconds = sentences[1:]
            else:
                firsts = sentences[:-1:2]
                seconds = sentences[1::2]
            return {
                'first': firsts,
                'second': seconds,
            }

        def example_len(x):
            return tf.math.minimum(
                tf.strings.length(x['first']), tf.strings.length(x['second']))

        # Split by lines.
        dataset = dataset.map(lambda x: x[text_key], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = split_by_lines(dataset)

        # Get pairs of neighboring sentences.
        dataset = dataset.map(split_into_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()

        # Remove examples with empty strings.
        dataset = dataset.filter(lambda x: example_len(x) > 0)
        return dataset


    def _buil_examples_from_files(self, files, label='nsp: ', label_sentences=False):
        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            sentence1_label, sentence2_label = '', ''
            if label_sentences:
                sentence1_label, sentence2_label = 'sentence1: ', 'sentence2: '

            og_dataset = tf.data.Dataset.from_tensor_slices({'text': [text]})
            empty = tf.constant('', dtype=tf.string, shape=[1])
            dataset = self.neighboring_pairs(og_dataset, text_key='text')
            # dataset_length = [i for i, _ in enumerate(dataset)][-1] + 1
            # print(dataset_length)
            dataset = dataset.shuffle(100000).batch(2, drop_remainder=True)

            def some_are_empty(*tensors):
                """See if at least one tensor has shape [0]."""
                empty = [tf.equal(tf.size(t), 0) for t in tensors]
                return tf.reduce_any(empty)

            def my_fn(x):
                """Function to be applied to each example in dataset."""
                negative_sampling = tf.random.uniform(shape=[]) < 0.5

                if self.nsp_generate:
                    def get_generated_sentence(sentence):
                        # you should decode bytes type to string type
                        generated_sentences = []
                        for sent in sentence.numpy():
                            encoded_prompt = self.gpt_tokenizer.encode(sent.decode('utf-8'), add_special_tokens=False, return_tensors="pt")
                            encoded_prompt = encoded_prompt.to(self.device)
                            if encoded_prompt.size()[-1] == 0:
                                input_ids = None
                            else:
                                input_ids = encoded_prompt

                            output_sequences = self.gpt_model.generate(
                                input_ids=input_ids,
                                max_length= 30 + len(encoded_prompt[0]),
                                temperature=1.0,
                                top_k=0,
                                top_p=0.9,
                                do_sample=True,
                                num_return_sequences=1,
                            )

                            # Remove the batch dimension when returning multiple sequences
                            if len(output_sequences.shape) > 2:
                                output_sequences.squeeze_()

                            generated_sequences = []

                            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                                generated_sequence = generated_sequence.tolist()
                                # Decode text
                                text = self.gpt_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                                total_sequence = text[len(self.gpt_tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
                                generated_sequences.append(total_sequence)
                            generated_sentences.append(tf.convert_to_tensor(generated_sequences[0], dtype=tf.string))
                        return tf.stack(generated_sentences)

                    encode_sentence = tf.py_function(get_generated_sentence, [x['first']], [tf.string])[0]
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        with sess.as_default():
                            encode_sentence.set_shape(x['first'].get_shape())

                    firsts, seconds = tf.cond(
                        negative_sampling,
                        lambda: (x['first'], x['second']),
                        lambda: (x['first'], encode_sentence),
                    )
                else:
                    firsts, seconds = tf.cond(
                        negative_sampling,
                        lambda: (x['first'], x['second']),
                        lambda: (x['first'], tf.stack([x['second'][1], x['second'][0]])),
                    )

                relation_label = tf.cond(
                    negative_sampling,
                    lambda: 'next',
                    lambda: 'not_next',
                )

                inputs = []
                for i in range(2):
                    first_inputs = firsts[i]
                    second_inputs = seconds[i]

                    def create_examples(first_i=first_inputs, second_i=second_inputs):
                        return tf.strings.join([
                            label,
                            sentence1_label,
                            first_i,
                            ' ',
                            sentence2_label,
                            second_i,
                        ])

                    inpt = tf.cond(
                        some_are_empty(first_inputs, second_inputs),
                        lambda: empty,
                        create_examples,
                    )

                    inputs.append(tf.strings.strip(inpt))
                inputs = tf.reshape(inputs, [-1])
                targets = tf.reshape(2 * [relation_label], [-1])
                return {'inputs': inputs, 'targets': targets}

            dataset = dataset.map(my_fn)
            dataset = dataset.unbatch()

            def example_len(x):
                return tf.math.minimum(
                    tf.strings.length(x['inputs']), tf.strings.length(x['targets']))

            dataset = dataset.filter(lambda x: example_len(x) > 0)
            tmp_input = []
            tmp_target = []
            for data in dataset:
                tmp_input.append(data['inputs'].numpy().decode('utf-8'))
                tmp_target.append(data['targets'].numpy().decode('utf-8'))

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                tmp_input, max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                tmp_target, max_length=5, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            for input, attention in zip(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]):
                self.inputs.append(
                    {"input_ids": input, "attention_mask": attention}
                )
            for input, attention in zip(tokenized_targets["input_ids"], tokenized_targets["attention_mask"]):
                self.targets.append(
                    {"input_ids": input, "attention_mask": attention}
                )

