import re
import glob
import os
import json
import pandas as pd
import argparse

def extractValLoss(checkpoint_path):
    """Eg checkpoint path format: path_to_dir/checkpoint_epoch=4-val_loss=0.450662.ckpt"""

    val_loss = float(re.search('val_loss=(.+?).ckpt', checkpoint_path).group(1))
    return val_loss

def extractEpochNum(checkpoint_path):
    """Eg checkpoint path format: path_to_dir/checkpoint_epoch=4-val_loss=0.450662.ckpt"""

    epoch_num = int(re.search('epoch=(.+?)-val_loss', checkpoint_path).group(1))
    return epoch_num

def decodeFolderName(folder_name, param_decode_map):
    folder_split_list = folder_name.split("_")
    param_dict = {}
    for i in range(0, len(folder_split_list), 2):
        param_dict[param_decode_map[folder_split_list[i]]] = folder_split_list[i + 1]
    return param_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tuning_output_dir', type=str, required=True,
                        help='Root output directory of the run_hyperparameter_tuning util.')
    parser.add_argument('--param_file', type=str, required=True,
                        help='Hyperparameter file. Can be in ".json"')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file to save the summary in csv format')

    args = parser.parse_known_args()[0]
    print(args)

    param_decode_map = {}
    with open(args.param_file, "r") as f:
        p_dict = json.load(f)
        for key in p_dict.keys():
            key_short = "".join([x[0] for x in key.lower().split("_")])
            param_decode_map[key_short] = key

    metrics_summary = ""
    config_folder_list = sorted(os.listdir(args.tuning_output_dir))
    for config_folder in config_folder_list:
        finetune_folder = os.path.join(args.tuning_output_dir, config_folder, "finetuning")

        checkpoint_list = glob.glob(os.path.join(finetune_folder, "checkpoint_*.ckpt"))
        sorted_list = sorted(checkpoint_list, key=lambda x: extractValLoss(x.split("/")[-1]))
        checkpoint_path = sorted_list[0]

        val_loss = str(round(extractValLoss(checkpoint_path.split("/")[-1]), 5))
        stop_epoch = extractEpochNum(checkpoint_path.split("/")[-1])

        metrics_path = os.path.join(args.tuning_output_dir, config_folder, "evaluation", "metrics_output.txt")

        try:
            with open(metrics_path, "r") as f:
                accuracy = round(float(f.read().split()[-1])*100, 2)

            cur_param_dict = decodeFolderName(config_folder, param_decode_map)
            cur_param_dict["stop_epoch"] = stop_epoch
            cur_param_dict["val_loss"] = val_loss
            cur_param_dict["accuracy"] = accuracy

            metrics_summary += json.dumps(cur_param_dict) + "\n"
        except:
            pass

    df_results_summary = pd.read_json(metrics_summary, lines=True)
    df_results_summary.to_csv(args.output_file, index=False)
    print(df_results_summary)



