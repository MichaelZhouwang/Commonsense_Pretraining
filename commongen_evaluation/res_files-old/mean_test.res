/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_mean.test.src_alpha.out
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.30003 (95%-conf.int. 0.29700 - 0.30292)
1 ROUGE-1 Average_P: 0.61586 (95%-conf.int. 0.61069 - 0.62056)
1 ROUGE-1 Average_F: 0.39428 (95%-conf.int. 0.39077 - 0.39760)
---------------------------------------------
1 ROUGE-2 Average_R: 0.07185 (95%-conf.int. 0.06970 - 0.07415)
1 ROUGE-2 Average_P: 0.16098 (95%-conf.int. 0.15621 - 0.16600)
1 ROUGE-2 Average_F: 0.09662 (95%-conf.int. 0.09384 - 0.09962)
---------------------------------------------
1 ROUGE-L Average_R: 0.23735 (95%-conf.int. 0.23461 - 0.23998)
1 ROUGE-L Average_P: 0.48499 (95%-conf.int. 0.48039 - 0.48945)
1 ROUGE-L Average_F: 0.31144 (95%-conf.int. 0.30821 - 0.31454)

/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_mean.test.src_alpha.out
>> ROUGE-F(1/2/l): 39.43/9.66/31.14
ROUGE-R(1/2/l): 30.00/7.18/23.73

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.049 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 17212, 'guess': [9669, 8172, 6675, 5178], 'testlen': 9669, 'correct': [6697, 2030, 495, 127]}
ratio: 0.561759237741
Bleu_1: 0.317
Bleu_2: 0.190
Bleu_3: 0.107
Bleu_4: 0.061
computing METEOR score...
METEOR: 0.164
computing CIDEr score...
CIDEr: 0.506
computing SPICE score...
SPICE: 0.172
Coverage
System level Coverage: 55.70