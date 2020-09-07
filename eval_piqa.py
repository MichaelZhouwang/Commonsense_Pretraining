if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("evaluate script")
    parser.add_argument("--input_x", type=str, required=True)
    parser.add_argument("--input_y", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)

    args = parser.parse_args()

    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    labels = pd.read_csv(args.input_y, sep='\t', header=None).values.tolist()
    preds = pd.read_csv(args.prediction, sep='\t', header=None).values.tolist()
    print(accuracy_score(labels, preds))

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