import json
import os
import argparse
# import sys
# sys.path.append("/".join(os.getcwd().split("/")[:-1]) + "/")

def convertTextToParams(input_file):
    param_dict_seq = []
    with open(input_file, "r") as f:
        header = None
        for cur_line in f:
            if header is None:
                header = list(cur_line.split())
            else:
                cur_row = list(cur_line.split())
                cur_dict = {}
                for i in range(len(header)):
                    cur_dict[header[i]] = str(cur_row[i])
                param_dict_seq.append(cur_dict)

    return param_dict_seq


param_dict_seq_global = []
def computeAllParamCombinations(key_list, cur_key_idx, param_dict, cur_param_seq_dict):
    global param_dict_seq_global
    if cur_key_idx >= len(key_list):
        param_dict_seq_global.append(cur_param_seq_dict)
        return

    cur_key = key_list[cur_key_idx]
    for cur_val in param_dict[cur_key]:
        cur_param_seq_dict[cur_key] = str(cur_val)
        computeAllParamCombinations(key_list, cur_key_idx + 1, param_dict, cur_param_seq_dict.copy())

def convertJsonToParams(input_file):
    global param_dict_seq_global
    with open(input_file, "r") as f:
        param_dict = json.load(f)

    param_keys = list(param_dict.keys())
    computeAllParamCombinations(param_keys, 0, param_dict, {})
    return param_dict_seq_global

def convertDictToCmdArgs(input_dict):
    out_string = ""
    for key, val in input_dict.items():
        out_string += "--" + str(key) + " " + str(val) + " "
    return out_string

def createFolderNameFromParamDict(input_dict):
    out_string = ""
    for key, val in input_dict.items():
        key_split = "".join([x[0] for x in key.lower().split("_")])
        out_string += key_split + "_" + str(val).lower() + "_"

    return out_string[:-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune_file', type=str, required=True,
                        help='Finetuning file')
    parser.add_argument('--predict_file', type=str, required=True,
                        help='Prediction file')
    parser.add_argument('--eval_file', type=str, required=True,
                        help='Evaluation file')
    parser.add_argument('--param_file', type=str, required=True,
                        help='Hyperparameter file. Can be in ".json" or ".tsv" format')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of GPUs to use for computation')
    parser.add_argument('--gpu_nums', type=str, default="0",
                        help='GPU ids separated by "," to use for computation')
    parser.add_argument('--root_output_dir', type=str, default="temp",
                        help='Root output directory')
    args = parser.parse_known_args()[0]

    if args.param_file.endswith(".json"):
        param_dict_seq = convertJsonToParams(args.param_file)
    else:
        param_dict_seq = convertTextToParams(args.param_file)

    for cur_param_seq in param_dict_seq:
        print("Running configuration:")
        print(cur_param_seq, "\n")

        cur_output_folder = os.path.join(args.root_output_dir, createFolderNameFromParamDict(cur_param_seq))

        # Finetuning:
        finetune_out_dir = os.path.join(cur_output_folder, 'finetuning')
        finetune_cmd = "python3 " + args.finetune_file + " " + convertDictToCmdArgs(cur_param_seq) + "--n_gpu " + str(args.n_gpu)\
                       + " --gpu_nums " + str(args.gpu_nums) + " --output_dir " + finetune_out_dir
        print(finetune_cmd, "\n")
        os.system(finetune_cmd)

        # Inference:
        predict_out_dir = os.path.join(cur_output_folder, 'prediction')
        predict_cmd = "python3 " + args.predict_file + " " + convertDictToCmdArgs(cur_param_seq) + " --checkpoint_dir " +\
                      finetune_out_dir + " --output_dir " + predict_out_dir
        print(predict_cmd, "\n")
        os.system(predict_cmd)

        # Evaluation:
        eval_out_dir = os.path.join(cur_output_folder, 'evaluation')
        eval_cmd = "python3 " + args.eval_file + " " + "--predicted_labels_dir " + predict_out_dir + " --output_dir " +\
                   eval_out_dir
        print(eval_cmd, "\n")
        os.system(eval_cmd)
        print("\n")





