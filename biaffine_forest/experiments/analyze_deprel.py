'''
for quick analysis
find accuracy by depedency relation label
'''

import csv
from pprint import pprint
from collections import defaultdict, Counter

with open('temp.conllu',mode='r', encoding='utf-8') as f:

    reader = csv.reader(f,delimiter='\t')

    cnt = 0
    flat = []
    l = []
    tmp=[]
    for row in reader:
        if row!=[]:
            tmp.append(row)
            flat.append(row)
        else:
            l.append(tmp)
            tmp=[]
    
    all_deprel_list = []
    error_deprel_list = []
    positive_deprel_list = []

    for i,lines in enumerate(l):
        for line in lines:
            correct_upos, _, correct_head_id, correct_deprel = line[4:8]
            predicted_head_id, predicted_deprel = line[8], line[9]

            all_deprel_list.append(correct_deprel)
            if (correct_head_id != predicted_head_id):
                error_deprel_list.append(correct_deprel)
            else:
                positive_deprel_list.append(correct_deprel)

            ## for analysis: print incorrect head for advcl relation
            if (correct_head_id != predicted_head_id) and correct_deprel=='advcl':
                print(i, line[1])
                #print(correct_head_id, predicted_head_id)
                #print(correct_deprel, predicted_deprel)
            #if (correct_head_id != predicted_head_id):
                #print(i, line[1], correct_upos)

    ## find deprel count
    c_all = Counter(all_deprel_list)
    pprint(c_all)
    c_pos = Counter(positive_deprel_list)
    #pprint(c_pos)

    ## find accuracy
    d = defaultdict()
    for deprel in c_all:
        #print(deprel)
        total = c_all[deprel]
        #print(total)
        if deprel in c_pos:
            correct = c_pos[deprel]
            acc = correct/total
            d[deprel] = acc
        else:
            acc = 0
            d[deprel] = acc
    
    d2 = sorted(d.items(), key=lambda x:x[1], reverse=True)
    pprint(d2)

    print(len(l))
    print(len(flat))

    ##
    #pprint(l[28])

