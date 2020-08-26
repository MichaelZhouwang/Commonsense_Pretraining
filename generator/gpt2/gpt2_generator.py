import json
import os
import warnings

import numpy as np

import tensorflow as tf
from generator.gpt2.src import encoder, model, sample

warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def standardize_punctuation(text):
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text

def cut_trailing_quotes(text):
    num_quotes = text.count('"')
    if num_quotes % 2 is 0:
        return text
    else:
        final_ind = text.rfind('"')
        return text[:final_ind]


def split_first_sentence(text):
    first_period = text.find(".")
    first_exclamation = text.find("!")

    if first_exclamation < first_period and first_exclamation > 0:
        split_point = first_exclamation + 1
    elif first_period > 0:
        split_point = first_period + 1
    else:
        split_point = text[0:20]

    return text[0:split_point], text[split_point:]


def cut_trailing_action(text):
    lines = text.split("\n")
    last_line = lines[-1]
    if (
        "you ask" in last_line
        or "You ask" in last_line
        or "you say" in last_line
        or "You say" in last_line
    ) and len(lines) > 1:
        text = "\n".join(lines[0:-1])
    return text


def cut_trailing_sentence(text):
    text = standardize_punctuation(text)
    last_punc = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punc <= 0:
        last_punc = len(text) - 1

    et_token = text.find("<")
    if et_token > 0:
        last_punc = min(last_punc, et_token - 1)

    act_token = text.find(">")
    if act_token > 0:
        last_punc = min(last_punc, act_token - 1)

    text = text[:last_punc+1]

    text = cut_trailing_quotes(text)
    text = cut_trailing_action(text)
    return text


class GPT2Generator:
    def __init__(self, generate_num=60, temperature=0.4, top_k=40, top_p=0.9, censor=True, force_cpu=False):
        self.generate_num = generate_num
        self.temp = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.censor = censor

        self.model_name = "345M"
        self.model_dir = "generator/gpt2/models"
        self.checkpoint_path = os.path.join(self.model_dir, self.model_name)

        models_dir = os.path.expanduser(os.path.expandvars(self.model_dir))
        self.batch_size = 1
        self.samples = 1

        self.enc = encoder.get_encoder(self.model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, self.model_name, "hparams.json")) as f:
            hparams.override_from_dict(json.load(f))
        seed = np.random.randint(0, 100000)

        config = None
        if force_cpu:
            config = tf.compat.v1.ConfigProto(
                device_count={"GPU": 0}
            )
        else:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        # np.random.seed(seed)
        # tf.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams,
            length=self.generate_num,
            context=self.context,
            batch_size=self.batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, self.model_name))
        saver.restore(self.sess, ckpt)

    def prompt_replace(self, prompt):
        # print("\n\nBEFORE PROMPT_REPLACE:")
        # print(repr(prompt))
        if len(prompt) > 0 and prompt[-1] == " ":
            prompt = prompt[:-1]

        # prompt = second_to_first_person(prompt)

        # print("\n\nAFTER PROMPT_REPLACE")
        # print(repr(prompt))
        return prompt

    def result_replace(self, result):
        # print("\n\nBEFORE RESULT_REPLACE:")
        # print(repr(result))

        result = cut_trailing_sentence(result)
        if len(result) == 0:
            return ""
        first_letter_capitalized = result[0].isupper()
        result = result.replace('."', '".')
        result = result.replace("#", "")
        result = result.replace("*", "")
        result = result.replace("\n\n", "\n")
        # result = first_to_second_person(result)

        if not first_letter_capitalized:
            result = result[0].lower() + result[1:]

        #
        # print("\n\nAFTER RESULT_REPLACE:")
        # print(repr(result))

        return result

    def generate_raw(self, prompt):
        context_tokens = self.enc.encode(prompt)
        generated = 0
        for _ in range(self.samples // self.batch_size):
            out = self.sess.run(
                self.output,
                feed_dict={
                    self.context: [context_tokens for _ in range(self.batch_size)]
                },
            )[:, len(context_tokens) :]
            for i in range(self.batch_size):
                generated += 1
                text = self.enc.decode(out[i])
        return text

    def generate(self, prompt, options=None, seed=1):

        debug_print = False
        prompt = self.prompt_replace(prompt)

        if debug_print:
            print("******DEBUG******")
            print("Prompt is: ", repr(prompt))

        text = self.generate_raw(prompt)

        if debug_print:
            print("Generated result is: ", repr(text))
            print("******END DEBUG******")

        result = text
        result = self.result_replace(result)
        if len(result) == 0:
            return self.generate(prompt)

        return result
