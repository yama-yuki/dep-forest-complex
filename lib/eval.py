import argparse
import csv
import json
import os, sys
import numpy as np
from collections import defaultdict, Counter
from pprint import pprint
from conllu import parse_incr

punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

class DepEvaluator:
    '''
    Measures UAS and LAS (without punctuation) of the output parsed dependency trees
    '''
    global punct

    def __init__(self,gold_path, pred_path):
        self._gold_path = gold_path
        self._pred_path = pred_path

    def _make_gold_dict(self):
        '''
        create a dict of sents(key) and gold trees(values)
        '''
        gold_d = defaultdict()
        with open(self._gold_path, mode='r', encoding='utf-8') as g:
            ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
            gold_conllu_list = self._make_conllu_list(g.readlines())

            print('total gold conllu: '+str(len(gold_conllu_list)))
            for gold_conllu in gold_conllu_list:
                tok_list = [line[1].rstrip('\n') for line in gold_conllu]
                sent = ' '.join(tok_list)
                if sent not in {' ',''}:
                    gold_d[sent] = gold_conllu
        
        ## some duplicates
        print(len(gold_d.keys()))
        print(sum([len(key.split(' ')) for key in gold_d.keys()])/len(gold_d.keys()))
        
        return gold_d

    def _make_conllu_list(self, lines):
        '''for forest
        read lines from rescored.conllu and return as conllu_list
        '''
        conllu_list = []
        tmp = []
        for line in lines:
            if line=='\n' or line[0]=='#':
                if tmp!=[]:
                    conllu_list.append(tmp)
                tmp = []
                continue
            sp = line.rstrip('\n').split('\t')
            tmp.append(sp)
        if tmp:
            conllu_list.append(tmp)
        cnt = sum([len(conllu) for conllu in conllu_list])

        print('len_conllu_list: '+str(len(conllu_list)))
        print('total_lines: '+str(cnt))

        return conllu_list

    def _read_nbest_json(self, parse_file):
        '''for nbest
        read parsed nbest.json file and return as conllu_list
        '''
        conllu_list = []

        with open(parse_file, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            nbests, sents = data[0], data[1]

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
        
        return conllu_list

    def _gold_parse(self):
        '''
        create a dict of sents(key) and gold trees(values)
        '''
        results = defaultdict()

        idx=0
        with open(self._gold_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                try:
                    tmp = [[tok['form'],tok['head'],tok['deprel']] for tok in tokenlist] ## tok, head, deprel                
                    results[idx] = tmp
                    idx+=1

                except:
                    sys.exit('hoge')
                
        return results

    def _complete_conllu_list(self, conllu_list, results):
        '''
        create output format with predictions and answers for debug purposes
        '''
        new_conllu_list = conllu_list
        for i,nbests in enumerate(new_conllu_list):
            correct_parse = results[i]
            for nbest in nbests:
                for j,tok in enumerate(nbest):
                    tok[6:8] = correct_parse[j][1:]

        return new_conllu_list

    def nbest(self):
        '''
        evaluator for nbest list rescoring approach
        '''
        pred_conllu_list = self._read_nbest_json(self._pred_path)

        ## ---(for analysis)---
        new_pred_conllu_list = self._complete_conllu_list(pred_conllu_list, self._gold_parse())
        ## write out 1best parsed tree (for debug)
        with open('temp.conllu',mode='w', encoding='utf-8') as o:
            writer = csv.writer(o,delimiter='\t')
            for nbests in new_pred_conllu_list:
                for nbest_parse in nbests[:1]:
                    for line in nbest_parse:
                        writer.writerow(line)
                    o.write('\n')
        ## ---(for analysis)---

        ## evaluate parsed trees from new_conllu_list
        correct = {'UAS': [], 'LAS': []}

        N = len(pred_conllu_list[0])
        for nbest_conllu in pred_conllu_list:
            for line in nbest_conllu[:1][0]:
                if len(line) == 10 and line[3] not in punct:
                    ## init with 0
                    correct['UAS'].append(0)
                    correct['LAS'].append(0)
                    ## replace with 1 if correct
                    ## check unlabeled attachment
                    if line[6] == line[8]:
                        correct['UAS'][-1] = 1
                    ## check labeled attachment
                    if line[6] == line[8] and line[7] == line[9]:
                        correct['LAS'][-1] = 1

                correct = {k:np.array(v) for k, v in correct.items()}

        UAS = (np.mean(correct['UAS']))*100
        LAS = (np.mean(correct['LAS']))*100
        c1, c2 = Counter(correct['UAS']), Counter(correct['LAS'])

        print('Head: '+str(c1), 'Head&Label: '+str(c2))
        
        return UAS, LAS

    def forest(self, mode1='wo_punct', mode2='vanilla'):
        '''
        evaluator for forest rescoring approach
        '''
        correct = {'UAS': [], 'LAS': []}
        gold_d = self._make_gold_dict()

        wrongs = []
        wrong_conllu_list = []
        cnt_conllu=0
        cnt_line=0

        with open(self._pred_path, mode='r', encoding='utf-8') as p:
            pred_conllu_list = self._make_conllu_list(p.readlines())
            for i,pred_conllu in enumerate(pred_conllu_list):
                tmp_sent = []
                for line in pred_conllu:
                    tmp_sent.append(line[1].rstrip('\n'))

                sent = ' '.join(tmp_sent)
                if sent!=' ' and  sent!='':
                    gold_conllu = gold_d[sent]
                    cnt_conllu+=1

                    flag=False
                    for pred,gold in zip(pred_conllu,gold_conllu):
                        '''
                        NOTE: 
                        original parser writes predictions in columns [6:8]
                        while in PTB, correct head&label are annotated in columns [8:10]
                        -> gold[6:8] is relocated to pred[8:10]
                        e.g.
                        ## pred (parser) ['17', 'assuming', '_', 'VERB', 'VBG', '_', '8', 'xcomp', '3', 'prep']
                        predictions are slotted in [6:8]
                        ## pred (forest) [17	assuming	assuming	VERB	_	_	_	_	8	xcomp]
                        predictions are slotted in [8:10]
                        ## gold ['17', 'assuming', '_', 'VERB', 'VBG', '_', '3', 'prep', '_', '_']
                        annotations are written in [6:8]
                        '''
                        cnt_line+=1

                        if mode2=='1best': ## modified
                            ## predicted
                            phead,plabel = pred[6:8]
                            ## correct answer
                            ghead,glabel = gold[6:8]
                        else:
                            ## predicted
                            phead,plabel = pred[8:10]
                            ## correct answer
                            ghead,glabel = gold[6:8]                           

                        ## check validity of each line
                        if mode1=='wo_punct':
                            valid = bool(len(pred) == 10 and gold[3] not in punct)
                        else:
                            valid = bool(len(pred) == 10)

                        ## give 1 for match / 0 for unmatch
                        if valid:
                            ## init with 0
                            correct['UAS'].append(0)
                            correct['LAS'].append(0)
                            ## replace with 1 if correct
                            ## check unlabeled attachment
                            if ghead == phead:
                                correct['UAS'][-1] = 1
                            ## check labeled attachment
                            if ghead == phead and glabel == plabel:
                                correct['LAS'][-1] = 1
                            ## if this conllu has an error
                            else:
                                flag=True
                                wrongs.append((ghead, phead, glabel, plabel))

                    if flag==True:
                        wrong_conllu_list.append(pred_conllu)

        correct = {k:np.array(v) for k, v in correct.items()}

        UAS = (np.mean(correct['UAS']))*100
        LAS = (np.mean(correct['LAS']))*100
        c1, c2 = Counter(correct['UAS']), Counter(correct['LAS'])

        ## total number of examples
        print('total conllus: '+str(cnt_conllu))
        print('total scored lines: '+str(len(correct['UAS'])))
        print('total processed lines: '+str(cnt_line))
        print('Head: '+str(c1), 'Head&Label: '+str(c2))

        return UAS, LAS, wrongs, wrong_conllu_list

'''
def make_conllu_list(lines):

    conllu_list = []
    tmp = []
    for line in lines:
        if line=='\n' or line[0]=='#':
            if tmp!=[]:
                conllu_list.append(tmp)
            tmp = []
            continue
        sp = line.rstrip('\n').split('\t')
        tmp.append(sp)
    if tmp:
        conllu_list.append(tmp)

    print('len_conllu_list: '+str(len(conllu_list)))
    cnt=0
    for conllu in conllu_list:
        for c in conllu:
            cnt+=1
            #print(c)
    print('total_lines: '+str(cnt))
    return conllu_list

def pred_evaluate(pred_path, test_d, mode='wo_punct'):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}

    wrongs = []
    wrong_conlls = []

    cnt_conllu=0
    cnt_line=0

    with open(pred_path, mode='r', encoding='utf-8') as p:
        pred_conllu_list = make_conllu_list(p.readlines())
        for i,pred_conllu in enumerate(pred_conllu_list):
            tmp_sent = []
            for line in pred_conllu:
                tmp_sent.append(line[1].rstrip('\n'))

            sent = ' '.join(tmp_sent)
            if sent!=' ' and  sent!='':
                test_conllu = test_d[sent]
                cnt_conllu+=1

                flag=False
                for pred,test in zip(pred_conllu,test_conllu):
                    cnt_line+=1
                    ##predicted
                    phead,plabel = pred[8:10]
                    ##correct answer
                    thead,tlabel = test[6:8]

                    if mode=='wo_punct':
                        if len(pred) == 10 and test[3] not in punct:
                            correct['UAS'].append(0)
                            correct['LAS'].append(0)

                            if thead == phead:
                                correct['UAS'][-1] = 1
                            if thead == phead and tlabel == plabel:
                                correct['LAS'][-1] = 1
                            else:
                                flag=True
                                wrongs.append((thead, phead, tlabel, plabel))
                    else:
                        if len(pred) == 10:
                            correct['UAS'].append(0)
                            correct['LAS'].append(0)

                            if thead == phead:
                                correct['UAS'][-1] = 1
                            if thead == phead and tlabel == plabel:
                                correct['LAS'][-1] = 1
                            else:
                                flag=True
                                wrongs.append((thead, phead, tlabel, plabel))

                if flag==True:
                    wrong_conlls.append(pred_conllu)

    correct = {k:np.array(v) for k, v in correct.items()}

    UAS = (np.mean(correct['UAS']))*100
    LAS = (np.mean(correct['LAS']))*100
    
    ## actual number of examples
    print('total pred conllus: '+str(cnt_conllu))
    print('total conllu lines: '+str(len(correct['UAS'])))
    print(cnt_line)
    c1 = Counter(correct['UAS'])
    print(c1)
    c2 = Counter(correct['LAS'])
    print(c2)

    return UAS, LAS, wrongs, wrong_conlls

def self_evaluate(test_path):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])
    correct = {'UAS': [], 'LAS': []}
    wrongs = []
    wrong_conlls = []

    with open(test_path, mode='r', encoding='utf-8') as t:
        ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
        test_conllu_list = make_conllu_list(t.readlines())

        for test_conllu in test_conllu_list:
            flag=False
            for test in test_conllu:
                ##predicted
                phead,plabel = test[8:10]
                ##correct answer
                thead,tlabel = test[6:8]                
                if len(test) == 10 and test[3] not in punct:
                    correct['UAS'].append(0)
                    correct['LAS'].append(0)

                    if thead == phead:
                        correct['UAS'][-1] = 1
                    if thead == phead and tlabel == plabel:
                        correct['LAS'][-1] = 1
                    else:
                        flag=True
                        wrongs.append((thead, phead, tlabel, plabel))
        
            if flag==True:
                wrong_conlls.append(test_conllu)

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

    return UAS, LAS, wrongs, wrong_conlls
'''

def analyze_deprel(wrongs):
    wrong_label = [wrong[2] for wrong in wrongs]
    #print(wrong_label)
    c = Counter(wrong_label)
    print(c)
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='file path for predictions')
    parser.add_argument('--biaf_path', help='file path for 1best data')
    parser.add_argument('--gold_path', help='file path for gold data')
    parser.add_argument('--plist', nargs='+', type=str, help='a list of file paths to evaluate')
    parser.add_argument('--nbest', action='store_true', help='evaluate nbest list')
    parser.add_argument('--eisner_k', type=str, help='EisnerK')

    args = parser.parse_args()
    pred_path = args.pred_path
    biaf_path = args.biaf_path
    gold_path = args.gold_path
    plist = args.plist
    eval_nbest=args.nbest
    eisner_k = args.eisner_k

    ## -----
    print('-----EVALUATION-----')
    if eval_nbest:
        print('---nbest---')
        evaluator = DepEvaluator(gold_path,pred_path)
        UAS, LAS = evaluator.nbest()
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
    
    else:
        print('---biaffine 1best---')
        biaf_evaluator = DepEvaluator(gold_path, biaf_path)
        UAS, LAS, wrongs, _ = biaf_evaluator.forest(mode1='wo_punct',mode2='1best')
        print('wo_punct')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
        analyze_deprel(wrongs)

        if plist:
            pass
        else:
            print('---vanilla---')
            vanilla_name = 'vanilla_'+eisner_k+'.conllu'
            dirname, basename = os.path.split(pred_path)
            vanilla_path = os.path.join(dirname, vanilla_name)
            print(vanilla_path)
            vanilla_evaluator = DepEvaluator(gold_path, vanilla_path)
            UAS, LAS, wrongs0, wrong_conlls0 = vanilla_evaluator.forest(mode1='wo_punct',mode2='vanilla')
            print('wo_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))
            analyze_deprel(wrongs0)

            print('---rescored---')
            print(pred_path)
            wo_punct_evaluator = DepEvaluator(gold_path, pred_path)
            UAS, LAS, wrongs1, wrong_conlls1 = wo_punct_evaluator.forest(mode1='wo_punct',mode2='rescored')
            print('wo_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))
            analyze_deprel(wrongs1)
            UAS, LAS, wrongs2, wrong_conlls2 = wo_punct_evaluator.forest(mode1='w_punct',mode2='rescored')
            print('w_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))    
            analyze_deprel(wrongs2)

    ## -----

    '''
    print('create a dict of correct trees')
    test_d = defaultdict()
    with open(gold_path, mode='r', encoding='utf-8') as t:
        ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
        test_conllu_list = make_conllu_list(t.readlines())

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
        UAS, LAS, wrongs, wrong_conlls = self_evaluate(gold_path)
        print('test file')
        print('1best Eisner')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
        #pprint(wrongs)
        print(analyze_deprel(wrongs))
        print('----------')
        print('pred file')
        print('Forest Decoder')

        if plist:
            for pred_path in plist:
                print('----------')
                UAS, LAS, _, _ = pred_evaluate(pred_path, test_d)
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
            vanilla_name = 'vanilla_4.conllu'
            dirname, basename = os.path.split(pred_path)
            vanilla_path = os.path.join(dirname, vanilla_name)
            print(vanilla_path)
            UAS, LAS, wrongs1, wrong_conlls1 = pred_evaluate(vanilla_path, test_d, mode='wo_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))
            analyze_deprel(wrongs1)
            UAS, LAS, wrongs1, wrong_conlls1 = pred_evaluate(vanilla_path, test_d, mode='w_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))
            analyze_deprel(wrongs1)
            print('-----')

            ##print('rescore_4.conllu')
            print(pred_path)
            UAS, LAS, wrongs2, wrong_conlls2 = pred_evaluate(pred_path, test_d, mode='wo_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))
            analyze_deprel(wrongs2)
            UAS, LAS, wrongs2, wrong_conlls2 = pred_evaluate(pred_path, test_d, mode='w_punct')
            print('UAS: '+str(UAS), 'LAS: '+str(LAS))    
            analyze_deprel(wrongs2)
            print('-----')
    '''

