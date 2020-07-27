/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt
/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.dev.base
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.47016 (95%-conf.int. 0.46490 - 0.47547)
1 ROUGE-1 Average_P: 0.52520 (95%-conf.int. 0.51950 - 0.53114)
1 ROUGE-1 Average_F: 0.46862 (95%-conf.int. 0.46439 - 0.47276)
---------------------------------------------
1 ROUGE-2 Average_R: 0.15837 (95%-conf.int. 0.15336 - 0.16319)
1 ROUGE-2 Average_P: 0.17048 (95%-conf.int. 0.16526 - 0.17611)
1 ROUGE-2 Average_F: 0.15332 (95%-conf.int. 0.14874 - 0.15791)
---------------------------------------------
1 ROUGE-L Average_R: 0.36528 (95%-conf.int. 0.36000 - 0.37022)
1 ROUGE-L Average_P: 0.40371 (95%-conf.int. 0.39890 - 0.40885)
1 ROUGE-L Average_F: 0.36200 (95%-conf.int. 0.35783 - 0.36592)

/home/bill/CommonGen-plus/methods/T5/transformer_local/examples/summarization/bart/t5.dev.base
>> ROUGE-F(1/2/l): 46.86/15.33/36.20
ROUGE-R(1/2/l): 47.02/15.84/36.53

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 1.732 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 11506, 'guess': [11385, 10392, 9399, 8406], 'testlen': 11385, 'correct': [6994, 3119, 1166, 404]}
ratio: 0.98948374761
Bleu_1: 0.608
Bleu_2: 0.425
Bleu_3: 0.281
Bleu_4: 0.180
computing METEOR score...
METEOR: 0.246
computing CIDEr score...
CIDEr: 0.973
computing SPICE score...
SPICE: 0.234
Coverage
System level Coverage: 83.77
