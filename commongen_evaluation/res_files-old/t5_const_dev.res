/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt
/home/bill/CommonGen-plus/methods/T5-DBA/transformer_local/examples/summarization/bart/t5.cons.dev
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.47346 (95%-conf.int. 0.46811 - 0.47896)
1 ROUGE-1 Average_P: 0.52536 (95%-conf.int. 0.51928 - 0.53107)
1 ROUGE-1 Average_F: 0.47254 (95%-conf.int. 0.46785 - 0.47702)
---------------------------------------------
1 ROUGE-2 Average_R: 0.17064 (95%-conf.int. 0.16497 - 0.17573)
1 ROUGE-2 Average_P: 0.18218 (95%-conf.int. 0.17642 - 0.18800)
1 ROUGE-2 Average_F: 0.16581 (95%-conf.int. 0.16087 - 0.17090)
---------------------------------------------
1 ROUGE-L Average_R: 0.37923 (95%-conf.int. 0.37417 - 0.38460)
1 ROUGE-L Average_P: 0.42062 (95%-conf.int. 0.41510 - 0.42657)
1 ROUGE-L Average_F: 0.37828 (95%-conf.int. 0.37356 - 0.38317)

/home/bill/CommonGen-plus/methods/T5-DBA/transformer_local/examples/summarization/bart/t5.cons.dev
>> ROUGE-F(1/2/l): 47.25/16.58/37.83
ROUGE-R(1/2/l): 47.35/17.06/37.92

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 1.788 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 11697, 'guess': [12332, 11339, 10346, 9355], 'testlen': 12332, 'correct': [6907, 3090, 1321, 612]}
ratio: 1.05428742413
Bleu_1: 0.560
Bleu_2: 0.391
Bleu_3: 0.269
Bleu_4: 0.189
computing METEOR score...
METEOR: 0.260
computing CIDEr score...
CIDEr: 0.917
computing SPICE score...
SPICE: 0.249
Coverage
System level Coverage: 86.21
