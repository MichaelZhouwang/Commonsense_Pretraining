merge_source = concept_source = keyword_source = option2_source = ""

with open('../datasets/concept_deshuffling/train.source') as fp:
    concept_source = fp.read()

with open('../datasets/keyword_lm/train.source') as fp:
    keyword_source = fp.read()

with open('../datasets/option2/train.source') as fp:
    option2_source = fp.read()

merge_source += concept_source
merge_source += "\n"
merge_source += keyword_source
merge_source += "\n"
merge_source += option2_source

with open('../datasets/mix/train.source', 'w') as fp:
    fp.write(merge_source)

merge_target = concept_target = keyword_target = option2_target = ""

with open('../datasets/concept_deshuffling/train.target') as fp:
    concept_target = fp.read()

with open('../datasets/keyword_lm/train.target') as fp:
    keyword_target = fp.read()

with open('../datasets/option2/train.target') as fp:
    option2_target = fp.read()

merge_target += concept_target
merge_target += "\n"
merge_target += keyword_target
merge_target += "\n"
merge_target += option2_target

with open('../datasets/mix/train.target', 'w') as fp:
    fp.write(merge_target)


merge_source = concept_source = keyword_source = option2_source = ""

with open('../datasets/concept_deshuffling/valid.source') as fp:
    concept_source = fp.read()

with open('../datasets/keyword_lm/valid.source') as fp:
    keyword_source = fp.read()

with open('../datasets/option2/valid.source') as fp:
    option2_source = fp.read()

merge_source += concept_source
merge_source += "\n"
merge_source += keyword_source
merge_source += "\n"
merge_source += option2_source

with open('../datasets/mix/valid.source', 'w') as fp:
    fp.write(merge_source)

merge_target = concept_target = keyword_target = option2_target = ""

with open('../datasets/concept_deshuffling/valid.target') as fp:
    concept_target = fp.read()

with open('../datasets/keyword_lm/valid.target') as fp:
    keyword_target = fp.read()

with open('../datasets/option2/valid.target') as fp:
    option2_target = fp.read()

merge_target += concept_target
merge_target += "\n"
merge_target += keyword_target
merge_target += "\n"
merge_target += option2_target

with open('../datasets/mix/valid.target', 'w') as fp:
    fp.write(merge_target)




