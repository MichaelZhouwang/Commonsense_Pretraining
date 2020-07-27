/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.dev.tgt.txt
/home/bill/CommonGen-plus/methods/BART/fairseq_local/bart.dev
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.56100 (95%-conf.int. 0.55550 - 0.56615)
1 ROUGE-1 Average_P: 0.54838 (95%-conf.int. 0.54328 - 0.55361)
1 ROUGE-1 Average_F: 0.54071 (95%-conf.int. 0.53634 - 0.54546)
---------------------------------------------
1 ROUGE-2 Average_R: 0.23343 (95%-conf.int. 0.22710 - 0.23996)
1 ROUGE-2 Average_P: 0.22219 (95%-conf.int. 0.21627 - 0.22815)
1 ROUGE-2 Average_F: 0.22129 (95%-conf.int. 0.21547 - 0.22722)
---------------------------------------------
1 ROUGE-L Average_R: 0.44768 (95%-conf.int. 0.44191 - 0.45350)
1 ROUGE-L Average_P: 0.43447 (95%-conf.int. 0.42935 - 0.43978)
1 ROUGE-L Average_F: 0.43015 (95%-conf.int. 0.42532 - 0.43521)

/home/bill/CommonGen-plus/methods/BART/fairseq_local/bart.dev
>> ROUGE-F(1/2/l): 54.07/22.13/43.02
ROUGE-R(1/2/l): 56.10/23.34/44.77

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 1.736 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 11625, 'guess': [11889, 10896, 9903, 8910], 'testlen': 11889, 'correct': [8162, 4047, 1973, 1000]}
ratio: 1.02270967742
Bleu_1: 0.687
Bleu_2: 0.505
Bleu_3: 0.370
Bleu_4: 0.275
computing METEOR score...
METEOR: 0.310
computing CIDEr score...
CIDEr: 1.412
computing SPICE score...
SPICE: 0.300
Coverage
System level Coverage: 97.56
