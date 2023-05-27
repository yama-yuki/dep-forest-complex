'''
finds 1-best UAS, LAS from n-bests
'''

import csv
import os
import sys
import json
import numpy as np

from pprint import pprint
from collections import defaultdict
from conllu import parse_incr

def evaluate(filename):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 10 and line[4] not in punct:
                print(line)
                correct['UAS'].append(0)
                correct['LAS'].append(0)
                if line[6] == line[8]:
                    correct['UAS'][-1] = 1
                ##if line[7] == line[9]:
                if line[6] == line[8] and line[7] == line[9]:
                    correct['LAS'][-1] = 1
        correct = {k:np.array(v) for k, v in correct.items()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

def evaluate_from_conllu_list(conllu_list):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    N = len(conllu_list[0])
    #print(lines_list)

    max_uas, max_las = [], []

    for nbests in conllu_list:
        uas_res, las_res = [], []

        for lines in nbests[:1]: ## <-
            correct = {'UAS': [], 'LAS': []}

            for line in lines:
                #line = line.strip().split('\t')
                if len(line) == 10 and line[4] not in punct:
                    correct['UAS'].append(0)
                    correct['LAS'].append(0)
                    if line[6] == line[8]:
                        correct['UAS'][-1] = 1
                    ##if line[7] == line[9]:
                    if line[6] == line[8] and line[7] == line[9]:
                        correct['LAS'][-1] = 1
            correct = {k:np.array(v) for k, v in correct.items()}
            uas, las = np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100
            uas_res.append(uas)
            las_res.append(las)

        #print(las_res)
        max_uas.append(max(uas_res))
        max_las.append(max(las_res))
    
    #print(max_uas)
    #print(max_las)
    print('UAS:'+str(np.mean(max_uas)), 'LAS: '+str(np.mean(max_las)))

def read_json(parse_file):
    conllu_list = []

    with open(parse_file, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        nbests, sents = data[0], data[1]

        '''
        print(sents[0])
        for parse in nbests[0]:
            pprint(parse)
        '''

        for i,(sent,nbest) in enumerate(zip(sents,nbests)):
            snt_len = range(len(sent))

            nbest_conllu_list = []
            for n in range(len(nbest)):
                d = defaultdict()
                tmp = []

                for edge in nbest[n]:
                    d[int(edge[1])] = [int(edge[2]),str(edge[3])]

                for i in snt_len:
                    ## 3	case	case	NOUN	NN	_	10	obl	_	_
                    word_idx = i+1
                    word = sent[i]
                    head_idx, deprel = d[word_idx]
                    result = [word_idx, word, '_', '_', '_', '_', '_', '_', head_idx, deprel]
                    
                    tmp.append(result)
                
                nbest_conllu_list.append(tmp)
            
            conllu_list.append(nbest_conllu_list)
    
    return conllu_list, sents

'''
for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
    tup = (
    i+1,
    word,
    self.tags[pred[3]] if pred[3] != -1 else self.tags[datum[2]],
    self.tags[pred[4]] if pred[4] != -1 else self.tags[datum[3]],
    str(pred[5]) if pred[5] != -1 else str(datum[4]),
    self.rels[pred[6]] if pred[6] != -1 else self.rels[datum[5]],
    str(pred[7]) if pred[7] != -1 else '_',
    self.rels[pred[8]] if pred[8] != -1 else '_',
    )
    f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
f.write('\n')
'''

def find_correct_parse_from_testset(test_file, sents):
    results = defaultdict()

    idx=0
    with open(test_file, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            test_sent = [tok['form'] for tok in tokenlist]
            try:
                tmp = [[tok['form'],tok['head'],tok['deprel']] for tok in tokenlist] ## tok, head, deprel                
                results[idx] = tmp
                idx+=1

            except:
                sys.exit('hoge')
            
    return results

def complete_conllu_list(conllu_list, results):
    new_conllu_list = conllu_list
    for i,nbests in enumerate(new_conllu_list):
        correct_parse = results[i]
        for nbest in nbests:
            for j,tok in enumerate(nbest):
                tok[6:8] = correct_parse[j][1:]
    return new_conllu_list

def main(parse_file, test_file):
    conllu_list, sents = read_json(parse_file)

    #print(len(sents))
    #conllu_list = conllu_list[:3]
    #sents = sents[:3]

    '''
    for i,(sent,conll) in enumerate(zip(sents,conllu_list)):
        print(sent)
        pprint(conll)
    
    for conll in conllu_list:
        pprint(conll[:1])
    '''

    results = find_correct_parse_from_testset(test_file, sents)

    new_conllu_list = complete_conllu_list(conllu_list, results)
    #pprint(new_conllu_list)

    write_temp_conllu(new_conllu_list)

    #print(str(len(new_conllu_list[0]))+'-Best')

    evaluate_from_conllu_list(new_conllu_list)

def write_temp_conllu(new_conllu_list):
    with open('temp.conllu',mode='w', encoding='utf-8') as o:
        writer = csv.writer(o,delimiter='\t')
        for nbests in new_conllu_list:
            for nbest_parse in nbests[:1]:
                for line in nbest_parse:
                    #print(line)
                    writer.writerow(line)
                o.write('\n')

if __name__ == '__main__':

    mode='file'
    #mode='dir'
    #mode='1best'

    file_name = 'test'
    #file_name = 'mytree_upos'
    
    test_file = 'data/wsj_sd_cophead/'+file_name+'.conllu'

    if mode=='file':
        '''
        target parse_file
        '''
        alpha='1'
        #parse_file = 'saves/ptb_cophead/'+file_name+'.conllu_10bestr'+str(alpha)+'_inside_vamodel_vvcond.json'
        #parse_file = 'saves/ptb_cophead/'+file_name+'.conllu_10bestr'+str(alpha)+'_inside_va.json'
        parse_file = 'saves/ptb_cophead/'+file_name+'.conllu_10bestr'+str(alpha)+'_inside.json'
        #parse_file = 'saves/ptb_cophead/'+file_name+'.conllu_10best.json'
        print(os.path.split(parse_file)[-1])
        main(parse_file, test_file)

    elif mode=='dir':
        dir_name = 'saves/ptb_cophead/'
        files = os.listdir(dir_name)
        files_file = [f for f in files  if os.path.isfile(os.path.join(dir_name, f)) if f.split('.')[0]==file_name]
        print(file_name)
        for f in sorted(files_file):
            print(f.split('.')[1][7:])
            main(os.path.join('saves','ptb_cophead',f), test_file)
            #print('')
    
    elif mode=='1best':
        parse_file = 'saves/ptb_cophead/'+file_name+'.conllu_1best.txt'
        #print(parse_file)
        print(evaluate(parse_file))

