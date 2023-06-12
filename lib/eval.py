import argparse
import sys
import numpy as np
from collections import defaultdict
from pprint import pprint

def make_conllu_list(lines):

    conllu_list = []
    tmp = []
    for line in lines:
        if line=='\n' or line[0]=='#':
            conllu_list.append(tmp)
            tmp = []
            continue
        sp = line.rstrip('\n').split('\t')
        tmp.append(sp)
    if tmp:
        conllu_list.append(tmp)

    return conllu_list

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

def pred_evaluate(pred_path, test_d):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}

    with open(pred_path, mode='r', encoding='utf-8') as p:
        pred_conllu_list = make_conllu_list(p.readlines())
        #print(pred_conllu_list)
        for i,pred_conllu in enumerate(pred_conllu_list):
            tmp_sent = []
            for line in pred_conllu:
                tmp_sent.append(line[1].rstrip('\n'))

            sent = ' '.join(tmp_sent)
            if sent!=' ' and  sent!='':
                test_conllu = test_d[sent]

                for pred,test in zip(pred_conllu,test_conllu):
                    ##predicted
                    phead,plabel = pred[8:10]
                    ##correct answer
                    thead,tlabel = test[6:8]

                    if len(pred) == 10 and test[4] not in punct:
                        correct['UAS'].append(0)
                        correct['LAS'].append(0)

                        if thead == phead:
                            correct['UAS'][-1] = 1
                        if thead == phead and tlabel == plabel:
                            correct['LAS'][-1] = 1

    correct = {k:np.array(v) for k, v in correct.items()}

    UAS = (np.mean(correct['UAS']))*100
    LAS = (np.mean(correct['LAS']))*100

    print(len(correct['UAS']))

    return UAS, LAS

def self_evaluate(test_path):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}

    with open(test_path, mode='r', encoding='utf-8') as t:
        ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
        test_conllu_list = make_conllu_list(t.readlines())
        #print(test_conllu_list)

        ## actual number of examples
        print(len(test_conllu_list))
        #pprint(test_conllu_list[:2])
        #print(test_conllu_list[:-10])
        for test_conllu in test_conllu_list:
            for test in test_conllu:
                ##predicted
                phead,plabel = test[8:10]
                ##correct answer
                thead,tlabel = test[6:8]                
                if len(test) == 10 and test[4] not in punct:
                    correct['UAS'].append(0)
                    correct['LAS'].append(0)

                    if thead == phead:
                        correct['UAS'][-1] = 1
                    if thead == phead and tlabel == plabel:
                        correct['LAS'][-1] = 1

    correct = {k:np.array(v) for k, v in correct.items()}

    UAS = (np.mean(correct['UAS']))*100
    LAS = (np.mean(correct['LAS']))*100

    print(len(correct['UAS']))

    return UAS, LAS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='file path for predictions')
    parser.add_argument('--test_path', help='file path for test data')
    args = parser.parse_args()
    pred_path = args.pred_path
    test_path = args.test_path

    print('creating a dict of correct trees')
    test_d = defaultdict()
    with open(test_path, mode='r', encoding='utf-8') as t:
        ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
        test_conllu_list = make_conllu_list(t.readlines())
        #print(test_conllu_list)

        ## actual number of examples
        print(len(test_conllu_list))
        #pprint(test_conllu_list[:2])
        #print(test_conllu_list[:-10])
        for test_conllu in test_conllu_list:
            tmp_sent = []
            for line in test_conllu:
                tmp_sent.append(line[1].rstrip('\n'))

            sent = ' '.join(tmp_sent)
            if sent!=' ' and  sent!='':
                #print(sent)
                test_d[sent] = test_conllu

        ## some duplicates
        print(len(test_d.keys()))
        print(sum([len(key.split(' ')) for key in test_d.keys()])/len(test_d.keys()))
    
        ## 1best eisner
        UAS, LAS = self_evaluate(test_path)
        print('test file')
        print('1best Eisner')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))        
        UAS, LAS = pred_evaluate(pred_path, test_d)
        print('pred file')
        print('Forest Decoder')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))


