/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test.base
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.44305 (95%-conf.int. 0.43924 - 0.44682)
1 ROUGE-1 Average_P: 0.54459 (95%-conf.int. 0.54012 - 0.54925)
1 ROUGE-1 Average_F: 0.46577 (95%-conf.int. 0.46275 - 0.46881)
---------------------------------------------
1 ROUGE-2 Average_R: 0.14164 (95%-conf.int. 0.13826 - 0.14528)
1 ROUGE-2 Average_P: 0.16926 (95%-conf.int. 0.16525 - 0.17357)
1 ROUGE-2 Average_F: 0.14569 (95%-conf.int. 0.14239 - 0.14918)
---------------------------------------------
1 ROUGE-L Average_R: 0.32913 (95%-conf.int. 0.32593 - 0.33254)
1 ROUGE-L Average_P: 0.40394 (95%-conf.int. 0.39992 - 0.40786)
1 ROUGE-L Average_F: 0.34551 (95%-conf.int. 0.34266 - 0.34843)

/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.test.base
>> ROUGE-F(1/2/l): 46.58/14.57/34.55
ROUGE-R(1/2/l): 44.30/14.16/32.91

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.100 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 19504, 'guess': [18036, 16539, 15042, 13545], 'testlen': 18036, 'correct': [11416, 4969, 1765, 612]}
ratio: 0.924733388023
Bleu_1: 0.583
Bleu_2: 0.402
Bleu_3: 0.260
Bleu_4: 0.164
computing METEOR score...
METEOR: 0.230
computing CIDEr score...
CIDEr: 0.916
computing SPICE score...
SPICE: 0.220
Coverage
System level Coverage: 76.67
