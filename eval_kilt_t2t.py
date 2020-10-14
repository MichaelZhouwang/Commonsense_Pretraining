import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
import os
import re
import string
from dataset import KILTT2TProcessor


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_match_score(y_true, y_preds):
    num_correct = 0
    for cur_y_true, cur_y_pred in zip(y_true, y_preds):
        num_correct += compute_exact(cur_y_true, cur_y_pred)
    em_score = num_correct / len(y_true)
    return em_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_labels_dir", type=str, default="datasets/kilt_ay2")
    parser.add_argument("--predicted_labels_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_known_args()[0]

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_type = args.ground_truth_labels_dir.split("/")[-1]
    kilt_proc = KILTT2TProcessor(task_type)
    val_samples = kilt_proc.get_dev_examples()

    predicted_labels_file = os.path.join(args.predicted_labels_dir, "dev.csv")
    output_file = os.path.join(args.output_dir, "metrics_output.txt")

    labels = []
    for sample in val_samples:
        labels.append(sample["output"])

    preds = pd.read_csv(predicted_labels_file, sep='\t', header=None).values.tolist()

    result_out = "Exact Match score = " + str(exact_match_score(labels, preds)) + "\n"
    print(result_out)
    with open(output_file, "w") as f:
        f.write(result_out)

    # stats = []
    # for _ in range(100):
    #     indices = [i for i in np.random.random_integers(0, len(preds) -1, size=len(preds))]
    #     stats.append(accuracy_score([labels[j] for j in indices], [preds[j] for j in indices]))
    #
    # alpha = 0.95
    # p = ((1.0 - alpha) /2.0) * 100
    # lower = max(0.0, np.percentile(stats, p))
    # p = (alpha +((1.0 - alpha) / 2.0)) * 100
    # upper = min(1.0, np.percentile(stats, p))
    # print(alpha * 100)
    # print("confidence interval :", lower * 100, upper * 100)
    # logger.info(f'{alpha * 10:.1f} confidence interval {lower * 100:.1f} and {upper * 100:.1f}, average: {np.mean(stats ) *100:.1f}')
