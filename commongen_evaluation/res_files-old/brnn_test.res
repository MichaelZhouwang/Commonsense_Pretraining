/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_brnn.test.src_alpha.out
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.28864 (95%-conf.int. 0.28552 - 0.29165)
1 ROUGE-1 Average_P: 0.49909 (95%-conf.int. 0.49422 - 0.50374)
1 ROUGE-1 Average_F: 0.35705 (95%-conf.int. 0.35351 - 0.36012)
---------------------------------------------
1 ROUGE-2 Average_R: 0.06083 (95%-conf.int. 0.05865 - 0.06306)
1 ROUGE-2 Average_P: 0.10946 (95%-conf.int. 0.10557 - 0.11306)
1 ROUGE-2 Average_F: 0.07609 (95%-conf.int. 0.07332 - 0.07859)
---------------------------------------------
1 ROUGE-L Average_R: 0.22480 (95%-conf.int. 0.22213 - 0.22743)
1 ROUGE-L Average_P: 0.38775 (95%-conf.int. 0.38331 - 0.39172)
1 ROUGE-L Average_F: 0.27789 (95%-conf.int. 0.27471 - 0.28078)

/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_brnn.test.src_alpha.out
>> ROUGE-F(1/2/l): 35.70/7.61/27.79
ROUGE-R(1/2/l): 28.86/6.08/22.48

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.138 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 17296, 'guess': [11729, 10232, 8735, 7238], 'testlen': 11729, 'correct': [7028, 1794, 424, 97]}
ratio: 0.678133672525
Bleu_1: 0.373
Bleu_2: 0.202
Bleu_3: 0.107
Bleu_4: 0.057
computing METEOR score...
METEOR: 0.158
computing CIDEr score...
CIDEr: 0.479
computing SPICE score...
SPICE: 0.150
Coverage
System level Coverage: 51.15
