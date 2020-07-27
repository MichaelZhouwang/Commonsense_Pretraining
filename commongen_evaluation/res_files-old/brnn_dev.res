/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt
/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_brnn.dev.src_alpha.out
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.31796 (95%-conf.int. 0.31348 - 0.32289)
1 ROUGE-1 Average_P: 0.49810 (95%-conf.int. 0.49239 - 0.50439)
1 ROUGE-1 Average_F: 0.37756 (95%-conf.int. 0.37300 - 0.38246)
---------------------------------------------
1 ROUGE-2 Average_R: 0.07698 (95%-conf.int. 0.07358 - 0.08057)
1 ROUGE-2 Average_P: 0.12534 (95%-conf.int. 0.12017 - 0.13053)
1 ROUGE-2 Average_F: 0.09227 (95%-conf.int. 0.08826 - 0.09636)
---------------------------------------------
1 ROUGE-L Average_R: 0.25765 (95%-conf.int. 0.25352 - 0.26200)
1 ROUGE-L Average_P: 0.40215 (95%-conf.int. 0.39706 - 0.40745)
1 ROUGE-L Average_F: 0.30568 (95%-conf.int. 0.30129 - 0.31018)

/home/bill/CommonGen-plus/methods/opennmt_based/model_output/commongen_brnn.dev.src_alpha.out
>> ROUGE-F(1/2/l): 37.76/9.23/30.57
ROUGE-R(1/2/l): 31.80/7.70/25.77

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 1.706 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 9763, 'guess': [7251, 6258, 5265, 4272], 'testlen': 7251, 'correct': [4353, 1232, 316, 87]}
ratio: 0.742702038308
Bleu_1: 0.425
Bleu_2: 0.243
Bleu_3: 0.136
Bleu_4: 0.078
computing METEOR score...
METEOR: 0.174
computing CIDEr score...
CIDEr: 0.604
computing SPICE score...
SPICE: 0.169
Coverage
System level Coverage: 58.95
