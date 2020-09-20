
# Automatic Hyper-parameter tuning experiments:

## CSQA:

1. To run the hyper-parameter tuning experiments, use the following command,

```python
python3 run_hyperparameter_tuning.py \
    --finetune_file finetune_csqa.py \
    --predict_file predict_csqa.py \
    --eval_file eval_csqa.py \
    --param_file hyperparameter_tuning_utils/csqa_params_tuning.json \
    --root_output_dir outputs/<output_dir> \
    --finetune_checkpoint_dir outputs/<checkpoint_dir> \
    --n_gpu 4 \
    --gpu_nums 0,1,2,3
```

where, <br>
<output_dir> is the root output directory for storing the results of the each setting of hyper parameter tuning experiments.
Eg: csqa_tuning_from_<checkpoint_dir> <br>
<checkpoint_dir> is the checkpoint directory which contains the model checkpoint to load during the fine-tuning of downstream task.

2. To summarize the results of fine-tuning experiments to a csv file, use the following command,

```python
python3 summarize_hyperparameter_tuning_outputs.py \
    --tuning_output_dir outputs/<output_dir> \
    --param_file hyperparameter_tuning_utils/csqa_params_tuning.json \
    --output_file outputs/<results_output_file>.csv
```

where, <br>
<output_dir> is the root output directory that is used while runing the hyper-parameter tuning experiments. <br>
<results_output_file>.csv is the final csv file that summarizes the various tuning experiments result.

Similarly for other tasks, refer to the commands in the following sections, 

### PIQA:

```python
python3 run_hyperparameter_tuning.py \
    --finetune_file finetune_piqa.py \
    --predict_file predict_piqa.py \
    --eval_file eval_piqa.py \
    --param_file hyperparameter_tuning_utils/piqa_params_tuning.json \
    --root_output_dir outputs/<output_dir> \
    --finetune_checkpoint_dir outputs/<checkpoint_dir> \
    --n_gpu 4 \
    --gpu_nums 0,1,2,3
```

```python
python3 summarize_hyperparameter_tuning_outputs.py \
    --tuning_output_dir outputs/<output_dir> \
    --param_file hyperparameter_tuning_utils/piqa_params_tuning.json \
    --output_file outputs/<results_output_file>.csv
```

### OBQA with KB:

```python
python3 run_hyperparameter_tuning.py \
    --finetune_file finetune_openbookqa.py \
    --predict_file predict_openbookqa.py \
    --eval_file eval_openbookqa.py \
    --param_file hyperparameter_tuning_utils/obqa_with_kb_params_tuning.json \
    --root_output_dir outputs/<output_dir> \
    --finetune_checkpoint_dir outputs/<checkpoint_dir> \
    --n_gpu 4 \
    --gpu_nums 0,1,2,3
```

```python
python3 summarize_hyperparameter_tuning_outputs.py \
    --tuning_output_dir outputs/<output_dir> \
    --param_file hyperparameter_tuning_utils/obqa_with_kb_params_tuning.json \
    --output_file outputs/<results_output_file>.csv
```

### OBQA without KB:

```python
python3 run_hyperparameter_tuning.py \
    --finetune_file finetune_openbookqa.py \
    --predict_file predict_openbookqa.py \
    --eval_file eval_openbookqa.py \
    --param_file hyperparameter_tuning_utils/obqa_without_kb_params_tuning.json \
    --root_output_dir outputs/<output_dir> \
    --finetune_checkpoint_dir outputs/<checkpoint_dir> \
    --n_gpu 4 \
    --gpu_nums 0,1,2,3
```

```python
python3 summarize_hyperparameter_tuning_outputs.py \
    --tuning_output_dir outputs/<output_dir> \
    --param_file hyperparameter_tuning_utils/obqa_without_kb_params_tuning.json \
    --output_file outputs/<results_output_file>.csv
```

### ANLI:

```python
python3 run_hyperparameter_tuning.py \
    --finetune_file finetune_anli.py \
    --predict_file predict_anli.py \
    --eval_file eval_anli.py \
    --param_file hyperparameter_tuning_utils/anli_params_tuning.json \
    --root_output_dir outputs/<output_dir> \
    --finetune_checkpoint_dir outputs/<checkpoint_dir> \
    --n_gpu 4 \
    --gpu_nums 0,1,2,3
```

```python
python3 summarize_hyperparameter_tuning_outputs.py \
    --tuning_output_dir outputs/<output_dir> \
    --param_file hyperparameter_tuning_utils/anli_params_tuning.json \
    --output_file outputs/<results_output_file>.csv
```



