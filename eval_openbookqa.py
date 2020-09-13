import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_labels", type=str, required=True)
    parser.add_argument("--predicted_labels", type=str, required=True)

    args = parser.parse_args()

    LABELS = ['A', 'B', 'C', 'D']
    ground_truth_labels = []
    with open(args.ground_truth_labels, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            ground_truth_labels.append(LABELS.index(cur_dict["answerKey"]))

    predicted_labels = pd.read_csv(args.predicted_labels, sep='\t', header=None).values.tolist()
    for i in range(len(predicted_labels)):
        predicted_labels[i] = LABELS.index(predicted_labels[i][0])

    print("Accuracy score = ", accuracy_score(ground_truth_labels, predicted_labels))

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