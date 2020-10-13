# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import uuid
import tensorflow.compat.v1 as tf
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os, glob
import pickle

import logging
import numpy as np
import torch
from generator.gpt2.gpt2_generator import *
from generator.concept.concept_generator import *
from tqdm import tqdm
import tensorflow_datasets as tfds

# tf.config.experimental.set_visible_devices([], 'GPU')
# tf.enable_eager_execution()
# import warnings
# warnings.filterwarnings("ignore", category=Warning)

class NSPDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, nsp_generate=False, concept_generate=False, max_len=512):
        self.type_path = type_path
        self.file_path = os.path.join(data_dir)
        self.files = glob.glob("%s/wiki.%s.raw" % (self.file_path, type_path))

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.nsp_generate = nsp_generate
        if self.nsp_generate:
            self.generator = GPT2Generator(temperature=0.7)

        self.concept_generate = concept_generate
        if self.concept_generate:
            self.generator = ConceptGenerator()

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
        self._build_examples_from_files(self.files)

    def neighboring_pairs_test(self, dataset, text_key='text', reuse_sentences=True):
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

    def neighboring_pairs_train(self, dataset, text_key='text', reuse_sentences=True):
        def split_by_lines(dataset):
            """Splits text in dataset by line, removing empty lines."""

            def my_fn(text):
                lines = tf.strings.split([text], sep='\n\n').values
                return tf.strings.strip(lines)

            dataset = dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            return dataset.filter(lambda x: tf.strings.length(x) > 0)

        def split_into_pairs(line):
            """Split a given text example into pairs of neighboring sentences."""
            # TODO(mmatena): Use better sentence segmentation.
            sentences = tf.strings.strip(tf.strings.split([line], sep='\n').values)
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

    def _build_examples_from_files(self, files, label='is_next: ', label_sentences=False):
        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            sentence1_label, sentence2_label = '', ''
            if label_sentences:
                sentence1_label, sentence2_label = 'sentence1: ', 'sentence2: '

            og_dataset = tf.data.Dataset.from_tensor_slices({'text': [text]})
            empty = tf.constant('', dtype=tf.string, shape=[1])
            if self.type_path == 'train':
                dataset = self.neighboring_pairs_train(og_dataset, text_key='text')
            else:
                dataset = self.neighboring_pairs_test(og_dataset, text_key='text')

            dataset = dataset.shuffle(100000).batch(2, drop_remainder=True)
            dataset_length = [i for i, _ in enumerate(tfds.as_numpy(dataset))][-1] + 1
            print(dataset_length)

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
                            generated_sentence = self.generator.generate(sent.decode('utf-8'))
                            generated_sentences.append(tf.convert_to_tensor(generated_sentence, dtype=tf.string))
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
                elif self.concept_generate:
                    def get_generated_sentence(sentence):
                        # you should decode bytes type to string type
                        generated_sentences = []
                        for sent in sentence.numpy():
                            generated_sentence = self.generator.generate(sent.decode('utf-8'))
                            generated_sentences.append(tf.convert_to_tensor(generated_sentence, dtype=tf.string))
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
                    lambda: 'true',
                    lambda: 'false',
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

            for i, data in tqdm(enumerate(tfds.as_numpy(dataset))):
                tmp_input.append(data['inputs'].decode('utf-8'))
                tmp_target.append(data['targets'].decode('utf-8'))

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                tmp_input, max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                tmp_target, max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
            )

            for input, attention in zip(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]):
                self.inputs.append(
                    {"input_ids": input, "attention_mask": attention}
                )
            for input, attention in zip(tokenized_targets["input_ids"], tokenized_targets["attention_mask"]):
                self.targets.append(
                    {"input_ids": input, "attention_mask": attention}
                )

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_source_length=32, max_target_length=32):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
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
        self.inputs = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".source"),
                                       self.max_source_length)
        self.targets = self.encode_file(self.tokenizer, os.path.join(self.data_dir, self.type_path + ".target"),
                                        self.max_target_length)

    def encode_file(self, tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
        examples = []
        with open(data_path, "r") as f:
            for text in f.readlines():
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
                    truncation=True
                )
                examples.append(tokenized)
        return examples


class InputExample(object):
    """A single multiple choice question. Here "article" is optional"""

    def __init__(self, qid, question, answers, label, article=None):
        """Construct an instance."""
        self.qid = qid
        self.question = question
        self.answers = answers
        self.label = label
        self.article = article


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a JSON file."""
        with tf.gfile.Open(input_file, "r") as f:
            return json.load(f)

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a JSON Lines file."""
        with tf.gfile.Open(input_file, "r") as f:
            return [json.loads(ln) for ln in f]


class CommonsenseQAProcessor(DataProcessor):
    """Processor for the CommonsenseQA data set."""

    SPLITS = ['qtoken', 'rand']
    LABELS = ['A', 'B', 'C', 'D', 'E']

    TRAIN_FILE_NAME = 'train_{split}_split.jsonl'
    DEV_FILE_NAME = 'dev_{split}_split.jsonl'
    TEST_FILE_NAME = 'test_{split}_split_no_answers.jsonl'

    def __init__(self, split):
        if split not in self.SPLITS:
            raise ValueError('split must be one of {", ".join(self.SPLITS)}.')
        self.split = split

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME.format(split=self.split)
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME.format(split=self.split)
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME.format(split=self.split)

        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            qid = line['id']
            question = line['question']['stem']
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            label = self.LABELS.index(line.get('answerKey', 'A'))
            examples.append(InputExample(
                qid=qid,
                question=question,
                answers=answers,
                label=label))

        return examples


class CSQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = CommonsenseQAProcessor('rand')

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12345', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class PIQAProcessor(DataProcessor):
    """Processor for the PIQA data set."""

    LABELS = ['sol1', 'sol2']

    TRAIN_FILE_NAME = 'train.jsonl'
    TRAIN_LABEL_NAME = 'train-labels.lst'
    DEV_FILE_NAME = 'valid.jsonl'
    DEV_LABEL_NAME = 'valid-labels.lst'
    TEST_FILE_NAME = 'tests.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        train_label_name = self.TRAIN_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, train_label_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        dev_label_name = self.DEV_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, dev_label_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME

        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), None, 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, labels, set_type):
        examples = []
        if labels is not None:
            for qid, (line, label) in enumerate(zip(lines, labels)):
                context = ""
                question = line["goal"]
                choices = [line["sol1"], line["sol2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label))
        else:
            for qid, line in enumerate(lines):
                context = ""
                question = line["goal"]
                choices = [line["sol1"], line["sol2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                # label = fields.get('label', None)
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class PIQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = PIQAProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class ANLIProcessor(DataProcessor):
    """Processor for the ANLI data set."""

    LABELS = ['hyp1', 'hyp2']

    TRAIN_FILE_NAME = 'train.jsonl'
    TRAIN_LABEL_NAME = 'train-labels.lst'
    DEV_FILE_NAME = 'dev.jsonl'
    DEV_LABEL_NAME = 'dev-labels.lst'
    TEST_FILE_NAME = 'test.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        train_label_name = self.TRAIN_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, train_label_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        dev_label_name = self.DEV_LABEL_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)),
                                     self._read_jsonl(os.path.join(data_dir, dev_label_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), None, 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, labels, set_type):
        examples = []
        if labels is not None:
            for (line, label) in zip(lines, labels):
                context = ""
                qid = line["story_id"]
                question = line["obs1"] + " " + line["obs2"]
                choices = [line["hyp1"], line["hyp2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label - 1))
        else:
            for line in lines:
                context = ""
                qid = line["story_id"]
                question = line["obs1"] + " " + line["obs2"]
                choices = [line["hyp1"], line["hyp2"]]
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class ANLIDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = ANLIProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


class OBQAProcessor(DataProcessor):
    """Processor for the OpenBook QA (OBQA) data set."""

    LABELS = ['A', 'B', 'C', 'D']

    def __init__(self, use_KB):
        self.use_KB = use_KB
        if self.use_KB:
            self.TRAIN_FILE_NAME = 'train_with_retrieved_facts_datamine.jsonl'
            self.DEV_FILE_NAME = 'dev_with_retrieved_facts_datamine.jsonl'
            self.TEST_FILE_NAME = 'test_with_retrieved_facts_datamine.jsonl'
        else:
            self.TRAIN_FILE_NAME = 'train.jsonl'
            self.DEV_FILE_NAME = 'dev.jsonl'
            self.TEST_FILE_NAME = 'test.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            qid = line['id']
            question = line['question']['stem']
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            label = self.LABELS.index(line['answerKey'])

            if self.use_KB:
                article = line['question']['retrieved_facts_context']
            else:
                article = None

            examples.append(InputExample(
                qid=qid,
                question=question,
                answers=answers,
                label=label,
                article=article))

        return examples


class OBQADataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, use_KB=False):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.use_KB = use_KB

        self.inputs = []
        self.targets = []

        self.proc = OBQAProcessor(self.use_KB)

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('1234', example.answers)]
        options = " ".join(options)

        if not self.use_KB:
            input_ = "context: %s  options: %s </s>" % (input_, options)
        else:
            article = example.article
            input_ = "context: %s  options: %s  article: %s </s>" % (input_, options, article)

        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)


# KILT Tasks:
class KILTFEVERProcessor(DataProcessor):
    """Processor for the KILT FEVER data set."""

    LABELS = ['SUPPORTS', 'REFUTES']

    TRAIN_FILE_NAME = 'fever-train-kilt.jsonl'
    DEV_FILE_NAME = 'fever-dev-kilt.jsonl'
    TEST_FILE_NAME = 'fever-test_without_answers-kilt.jsonl'

    def get_train_examples(self, data_dir):
        train_file_name = self.TRAIN_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, train_file_name)), 'train')

    def get_dev_examples(self, data_dir):
        dev_file_name = self.DEV_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, dev_file_name)), 'dev')

    def get_test_examples(self, data_dir):
        test_file_name = self.TEST_FILE_NAME
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, test_file_name)), 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        if set_type != "test":
            for line in lines:
                context = ""
                qid = line["id"]
                question = line["input"]
                choices = self.LABELS
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                label = self.LABELS.index(line["output"][0]["answer"])
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=label))
        else:
            for line in lines:
                context = ""
                qid = line["id"]
                question = line["input"]
                choices = self.LABELS
                choices = [c + "." if not c.endswith(".") else c for c in choices]
                examples.append(InputExample(
                    qid=qid,
                    question=question,
                    answers=choices,
                    label=None))
        return examples


class KILTFEVERDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = KILTFEVERProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == "train":
            examples = self.proc.get_train_examples(self.data_dir)
        elif self.type_path == "valid":
            examples = self.proc.get_dev_examples(self.data_dir)
        else:
            examples = self.proc.get_test_examples(self.data_dir)

        for example in examples:
            self._create_features(example)

    def _create_features(self, example):
        input_ = example.question
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = " ".join(options)
        input_ = "context: %s  options: %s </s>" % (input_, options)
        target = "%s </s>" % str(int(example.label) + 1)

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt", truncation=True
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)
