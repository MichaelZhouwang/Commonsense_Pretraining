/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/T5-DBA/transformer_local/examples/summarization/bart/t5.cons.test
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.47491 (95%-conf.int. 0.47061 - 0.47900)
1 ROUGE-1 Average_P: 0.53860 (95%-conf.int. 0.53378 - 0.54324)
1 ROUGE-1 Average_F: 0.48057 (95%-conf.int. 0.47722 - 0.48404)
---------------------------------------------
1 ROUGE-2 Average_R: 0.16928 (95%-conf.int. 0.16508 - 0.17346)
1 ROUGE-2 Average_P: 0.18637 (95%-conf.int. 0.18198 - 0.19092)
1 ROUGE-2 Average_F: 0.16804 (95%-conf.int. 0.16412 - 0.17210)
---------------------------------------------
1 ROUGE-L Average_R: 0.36338 (95%-conf.int. 0.35918 - 0.36748)
1 ROUGE-L Average_P: 0.41064 (95%-conf.int. 0.40616 - 0.41537)
1 ROUGE-L Average_F: 0.36708 (95%-conf.int. 0.36347 - 0.37060)

/home/bill/CommonGen-plus/methods/T5-DBA/transformer_local/examples/summarization/bart/t5.cons.test
>> ROUGE-F(1/2/l): 48.06/16.80/36.71
ROUGE-R(1/2/l): 47.49/16.93/36.34

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.444 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 20280, 'guess': [20761, 19264, 17767, 16270], 'testlen': 20761, 'correct': [11870, 5317, 2286, 981]}
ratio: 1.02371794872
Bleu_1: 0.572
Bleu_2: 0.397
Bleu_3: 0.273
Bleu_4: 0.187
computing METEOR score...
METEOR: 0.253
computing CIDEr score...
CIDEr: 0.862
computing SPICE score...
SPICE: 0.243
Coverage
System level Coverage: 83.98
