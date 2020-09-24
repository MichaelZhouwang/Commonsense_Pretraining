# source ~/.bashrc
INPUT_FILE=~/Commonsense_Pretraining/commongen/test.source
TRUTH_FILE=~/Commonsense_Pretraining/commongen/test.target
PRED_FILE=$1

echo ${INPUT_FILE}
echo ${TRUTH_FILE}
echo ${PRED_FILE}

echo "Start running ROUGE"

cd ~/CommonGen/methods/unilm_based
/home/danny911kr/miniconda3/envs/unilm_env/bin/python unilm/src/gigaword/eval.py --pred ${PRED_FILE}   --gold ${TRUTH_FILE} --perl


echo "BLEU/METER/CIDER/SPICE"
cd ~/CommonGen/evaluation/Traditional/eval_metrics/
/home/danny911kr/miniconda3/envs/coco_score/bin/python eval.py --key_file ${INPUT_FILE} --gts_file ${TRUTH_FILE} --res_file ${PRED_FILE}


echo "PivotScore"
cd ~/CommonGen/evaluation/PivotScore
/home/danny911kr/miniconda3/envs/pivot_score/bin/python evaluate.py --pred ${PRED_FILE}   --ref ${TRUTH_FILE} --cs ${INPUT_FILE} --cs_str ~/CommonGen/dataset/final_data/commongen/commongen.test.cs_str.txt

echo "_________________"
echo "Correct BERTScore"
cd ~/CommonGen/evaluation/BERTScore
CUDA_VISIBLE_DEVICES=3 /home/danny911kr/miniconda3/envs/bert_score/bin/python evaluate.py --pred ${PRED_FILE}   --ref ${TRUTH_FILE} --cs ~/CommonGen/dataset/final_data/commongen/commongen.test.src_alpha.txt