/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_transformer.test.src_alpha.out
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.26619 (95%-conf.int. 0.26275 - 0.26929)
1 ROUGE-1 Average_P: 0.53511 (95%-conf.int. 0.52977 - 0.54077)
1 ROUGE-1 Average_F: 0.34686 (95%-conf.int. 0.34322 - 0.35036)
---------------------------------------------
1 ROUGE-2 Average_R: 0.06528 (95%-conf.int. 0.06290 - 0.06752)
1 ROUGE-2 Average_P: 0.13889 (95%-conf.int. 0.13437 - 0.14345)
1 ROUGE-2 Average_F: 0.08615 (95%-conf.int. 0.08325 - 0.08901)
---------------------------------------------
1 ROUGE-L Average_R: 0.21444 (95%-conf.int. 0.21167 - 0.21732)
1 ROUGE-L Average_P: 0.42921 (95%-conf.int. 0.42433 - 0.43434)
1 ROUGE-L Average_F: 0.27909 (95%-conf.int. 0.27588 - 0.28252)

/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_transformer.test.src_alpha.out
>> ROUGE-F(1/2/l): 34.69/8.62/27.91
ROUGE-R(1/2/l): 26.62/6.53/21.44

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.282 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 17222, 'guess': [10107, 8610, 7113, 5616], 'testlen': 10107, 'correct': [6304, 1891, 550, 150]}
ratio: 0.586865636976
Bleu_1: 0.309
Bleu_2: 0.183
Bleu_3: 0.109
Bleu_4: 0.064
computing METEOR score...
METEOR: 0.152
computing CIDEr score...
CIDEr: 0.454
computing SPICE score...
SPICE: 0.145
Coverage
System level Coverage: 49.10