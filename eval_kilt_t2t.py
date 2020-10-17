import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
import os
import re
import string
from dataset import KILTT2TProcessor
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _calculate_metrics(gold_records, guess_records):
    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0

    for guess_item, gold_item in zip(guess_records, gold_records):
        total_count += 1
        gold_candidate_answers = gold_item
        guess_answer = guess_item.strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )
        normalized_f1 += local_f1

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count

    return {
        "accuracy": accuracy,
        "em": normalized_em,
        "f1": normalized_f1,
    }


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
    val_samples = kilt_proc.get_dev_examples(args.ground_truth_labels_dir)

    predicted_labels_file = os.path.join(args.predicted_labels_dir, "dev.csv")
    output_file = os.path.join(args.output_dir, "metrics_output.txt")

    labels = []
    for sample in val_samples:
        labels.append(sample["output"])

    preds = []
    with open(predicted_labels_file, "r") as f:
        for cur_line in f:
            preds.append(cur_line.strip())

    result_out = json.dumps(_calculate_metrics(labels, preds))
    print(result_out)
    with open(output_file, "w") as f:
        f.write(result_out)
