/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.52292 (95%-conf.int. 0.51902 - 0.52677)
1 ROUGE-1 Average_P: 0.66379 (95%-conf.int. 0.65995 - 0.66774)
1 ROUGE-1 Average_F: 0.57309 (95%-conf.int. 0.56964 - 0.57642)
---------------------------------------------
1 ROUGE-2 Average_R: 0.20146 (95%-conf.int. 0.19712 - 0.20577)
1 ROUGE-2 Average_P: 0.25436 (95%-conf.int. 0.24947 - 0.25957)
1 ROUGE-2 Average_F: 0.22006 (95%-conf.int. 0.21575 - 0.22475)
---------------------------------------------
1 ROUGE-L Average_R: 0.39298 (95%-conf.int. 0.38902 - 0.39700)
1 ROUGE-L Average_P: 0.49570 (95%-conf.int. 0.49124 - 0.49987)
1 ROUGE-L Average_F: 0.42973 (95%-conf.int. 0.42584 - 0.43367)

/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test
>> ROUGE-F(1/2/l): 57.31/22.01/42.97
ROUGE-R(1/2/l): 52.29/20.15/39.30

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.124 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 18115, 'guess': [16739, 15242, 13745, 12248], 'testlen': 16739, 'correct': [12998, 6563, 3131, 1505]}
ratio: 0.924040850124
Bleu_1: 0.715
Bleu_2: 0.533
Bleu_3: 0.390
Bleu_4: 0.286
computing METEOR score...
METEOR: 0.301
computing CIDEr score...
CIDEr: 1.496
computing SPICE score...
SPICE: 0.316
Coverage
System level Coverage: 95.29
