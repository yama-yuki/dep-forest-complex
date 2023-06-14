import argparse
import os, sys
import numpy as np
from collections import defaultdict, Counter
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

def pred_evaluate(pred_path, test_d):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}

    wrongs = []

    cnt=0
    with open(pred_path, mode='r', encoding='utf-8') as p:
        pred_conllu_list = make_conllu_list(p.readlines())
        for i,pred_conllu in enumerate(pred_conllu_list):
            tmp_sent = []
            for line in pred_conllu:
                tmp_sent.append(line[1].rstrip('\n'))

            sent = ' '.join(tmp_sent)
            if sent!=' ' and  sent!='':
                test_conllu = test_d[sent]
                cnt+=1

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
                        else:
                            wrongs.append((thead, phead, tlabel, plabel))

    correct = {k:np.array(v) for k, v in correct.items()}

    UAS = (np.mean(correct['UAS']))*100
    LAS = (np.mean(correct['LAS']))*100
    
    ## actual number of examples
    print('total pred conllus: '+str(cnt))
    print('total conllu lines: '+str(len(correct['UAS'])))
    c1 = Counter(correct['UAS'])
    print(c1)
    c2 = Counter(correct['LAS'])
    print(c2)

    #print(wrongs)

    return UAS, LAS, wrongs

def analyze_deprel(wrongs):
    wrong_label = [wrong[2] for wrong in wrongs]
    #print(wrong_label)
    c = Counter(wrong_label)
    print(c)
    return

def self_evaluate(test_path):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}
    wrongs = []

    with open(test_path, mode='r', encoding='utf-8') as t:
        ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
        test_conllu_list = make_conllu_list(t.readlines())

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
                    else:
                        wrongs.append((thead, phead, tlabel, plabel))

    correct = {k:np.array(v) for k, v in correct.items()}

    UAS = (np.mean(correct['UAS']))*100
    LAS = (np.mean(correct['LAS']))*100

    ## actual number of examples
    print('total test conllus: '+str(len(test_conllu_list)))
    print('total conllu lines: '+str(len(correct['UAS'])))
    c1 = Counter(correct['UAS'])
    print(c1)
    c2 = Counter(correct['LAS'])
    print(c2)

    return UAS, LAS, wrongs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='file path for predictions')
    parser.add_argument('--test_path', help='file path for test data')
    parser.add_argument('--plist', nargs='+', type=str, help='a list of file paths to evaluate')
    args = parser.parse_args()
    pred_path = args.pred_path
    test_path = args.test_path
    plist = args.plist

    print('create a dict of correct trees')
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
    
        print('-----EVALUATION-----')
        ## 1best eisner
        UAS, LAS, wrongs = self_evaluate(test_path)
        print('test file')
        print('1best Eisner')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
        print(analyze_deprel(wrongs))
        print('----------')
        print('pred file')
        print('Forest Decoder')

        if plist:
            for pred_path in plist:
                print('----------')
                UAS, LAS, _ = pred_evaluate(pred_path, test_d)
                root, ext = os.path.splitext(pred_path)
                dirname, basename = os.path.split(root)
                params = basename.split('_')[-1].split('-')
                if len(params)>1:
                    k,a,b = params
                    print('K='+str(k)+', a='+str(a)+', b='+str(b)+', rescore')
                else:
                    k = params[0]
                    print('K='+str(k)+', vanilla')
                print('UAS: '+str(UAS), 'LAS: '+str(LAS))

        else:
            print('vanilla_4.conllu')
            UAS, LAS, wrongs1 = pred_evaluate('/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/mod/k4/vanilla_4.conllu', test_d)
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))

            print('rescore_4.conllu')
            UAS, LAS, wrongs2 = pred_evaluate(pred_path, test_d)
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))

            print('-----')
            analyze_deprel(wrongs1)
            print('-----')
            analyze_deprel(wrongs2)