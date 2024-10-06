import os, sys

from conll import pkl_analysis

## EisnerK
EisnerK=str(4)
data_type='test'
## path
PJ_DIR='/home/is/yuki-yama/work/d3/dep-forest-complex'
pkl_dir=os.path.join(PJ_DIR,'biaffine_forest','pkl',data_type,'k'+EisnerK)

print('1: Loading Parsed Forests')
all_forests,all_sents = pkl_analysis(pkl_dir)

tmp=[]
for i,(forest,sent) in enumerate(zip(all_forests,all_sents)):
    #print(len(sent))
    hes = forest['hyperedges']
    if len(sent) > 40:
        for he in hes:
            tmp.append(he["prob"])
    #print(len(forest['hyperedges']))

print(max(tmp))
print(min(tmp))

print('done')
