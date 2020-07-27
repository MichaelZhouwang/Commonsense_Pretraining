/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.src_alpha.txt
/home/bill/CommonGen-plus/dataset/final_data/commongen/commongen.test.tgt.txt
/home/bill/CommonGen-plus/methods/BART/fairseq_local/bart.test
Start running ROUGE
here!!!!!!!!!!!!!
1
---------------------------------------------
1 ROUGE-1 Average_R: 0.57482 (95%-conf.int. 0.57076 - 0.57917)
1 ROUGE-1 Average_P: 0.55743 (95%-conf.int. 0.55339 - 0.56126)
1 ROUGE-1 Average_F: 0.55279 (95%-conf.int. 0.54942 - 0.55602)
---------------------------------------------
1 ROUGE-2 Average_R: 0.23369 (95%-conf.int. 0.22888 - 0.23844)
1 ROUGE-2 Average_P: 0.22268 (95%-conf.int. 0.21798 - 0.22718)
1 ROUGE-2 Average_F: 0.22235 (95%-conf.int. 0.21787 - 0.22663)
---------------------------------------------
1 ROUGE-L Average_R: 0.43752 (95%-conf.int. 0.43307 - 0.44204)
1 ROUGE-L Average_P: 0.42196 (95%-conf.int. 0.41784 - 0.42612)
1 ROUGE-L Average_F: 0.41981 (95%-conf.int. 0.41590 - 0.42384)

/home/bill/CommonGen-plus/methods/BART/fairseq_local/bart.test
>> ROUGE-F(1/2/l): 55.28/22.23/41.98
ROUGE-R(1/2/l): 57.48/23.37/43.75

BLEU/METER/CIDER/SPICE
SPICE evaluation took: 2.221 s
tokenization...
setting up scorers...
computing Bleu score...
{'reflen': 20680, 'guess': [21331, 19834, 18337, 16840], 'testlen': 21331, 'correct': [14798, 7297, 3429, 1678]}
ratio: 1.03147969052
Bleu_1: 0.694
Bleu_2: 0.505
Bleu_3: 0.363
Bleu_4: 0.263
computing METEOR score...
METEOR: 0.309
computing CIDEr score...
CIDEr: 1.392
computing SPICE score...
SPICE: 0.306
Coverage
System level Coverage: 97.35
