/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt
/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.dev
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.52095 (95%-conf.int. 0.51577 - 0.52603)
1 ROUGE-1 Average_P: 0.63720 (95%-conf.int. 0.63166 - 0.64236)
1 ROUGE-1 Average_F: 0.56053 (95%-conf.int. 0.55580 - 0.56493)
---------------------------------------------
1 ROUGE-2 Average_R: 0.20474 (95%-conf.int. 0.19908 - 0.21074)
1 ROUGE-2 Average_P: 0.24959 (95%-conf.int. 0.24290 - 0.25675)
1 ROUGE-2 Average_F: 0.21981 (95%-conf.int. 0.21398 - 0.22633)
---------------------------------------------
1 ROUGE-L Average_R: 0.41338 (95%-conf.int. 0.40802 - 0.41885)
1 ROUGE-L Average_P: 0.50312 (95%-conf.int. 0.49733 - 0.50936)
1 ROUGE-L Average_F: 0.44411 (95%-conf.int. 0.43886 - 0.44957)

/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.dev
>> ROUGE-F(1/2/l): 56.05/21.98/44.41
ROUGE-R(1/2/l): 52.09/20.47/41.34

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 1.749 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 10468, 'guess': [9975, 8982, 7989, 6996], 'testlen': 9975, 'correct': [7610, 3891, 1899, 953]}
ratio: 0.952904088651
Bleu_1: 0.726
Bleu_2: 0.547
Bleu_3: 0.408
Bleu_4: 0.306
computing METEOR score...
METEOR: 0.310
computing CIDEr score...
CIDEr: 1.584
computing SPICE score...
SPICE: 0.318
Coverage
System level Coverage: 97.04
