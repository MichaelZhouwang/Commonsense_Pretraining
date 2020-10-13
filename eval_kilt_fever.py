import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_labels_dir", type=str, default="datasets/kilt_fever")
    parser.add_argument("--predicted_labels_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_known_args()[0]

    # Create a folder if output_dir doesn't exists:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ground_truth_labels_file = os.path.join(args.ground_truth_labels_dir, "fever-dev-kilt.jsonl")
    predicted_labels_file = os.path.join(args.predicted_labels_dir, "dev.csv")
    output_file = os.path.join(args.output_dir, "metrics_output.txt")

    labels = []
    with open(ground_truth_labels_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            labels.append(cur_dict["output"][0]["answer"])

    preds = pd.read_csv(predicted_labels_file, sep='\t', header=None).values.tolist()

    result_out = "Accuracy score = " + str(accuracy_score(labels, preds)) + "\n"
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