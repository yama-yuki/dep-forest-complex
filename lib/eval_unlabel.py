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

    def __init__(self,gold_path, pred_path, unlabel=False):
        self._gold_path = gold_path
        self._pred_path = pred_path
        self._unlabel = unlabel

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
    
    def _make_gold_dict_subord(self, lang):
        gold_d_list = [defaultdict(), defaultdict(), defaultdict(), defaultdict()]
        with open(self._gold_path, mode='r', encoding='utf-8') as g:
            ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']
            gold_conllu_list = self._make_conllu_list(g.readlines())
            print('_make_gold_dict_subord')
            print('total gold conllu: '+str(len(gold_conllu_list)))
            for gold_conllu in gold_conllu_list:
                tok_list = [line[1].rstrip('\n') for line in gold_conllu]
                sent = ' '.join(tok_list)
                count = self._count_subord2(gold_conllu,lang)

                if sent not in {' ',''}:
                    gold_d_list[count][sent] = gold_conllu

        return gold_d_list
    
    def _count_subord(self, gold_conllu, lang):
        count=0
        for line in gold_conllu:
            if lang=='en':
                if line[3]=='SCONJ':
                    count+=1
            elif lang=='ch':
                if line[3][:1]=='V':
                    count+=1
            else:
                sys.exit('SPECIFY: lang (en/ch)')
        if count>=3:
            count = 3
        return count

    def _count_subord2(self, gold_conllu, lang):
        count=0

        tmp=0
        for line in gold_conllu:
            if lang=='en':
                if line[3]=='VERB' and line[7] in {'advcl','ccomp','vmod','xcomp'}:
                    count+=1
            elif lang=='ch':
                if line[3][:1]=='V':
                    count+=1
            else:
                sys.exit('SPECIFY: lang (en/ch)')
        if count>=3:
            count = 3
        return count

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
                    tmp = [[tok['form'],tok['head'],tok['deprel'],tok['xpos']] for tok in tokenlist] ## tok, head, deprel                
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
                    tok[6:8] = correct_parse[j][1:3]

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
            nbest_conllu = nbest_conllu[0] ##1best out of 16best
            ##for line in nbest_conllu[:1][0]:
            for line in nbest_conllu:
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

        g_label_d = dict()
        cnt_root=0

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
                            if glabel in g_label_d.keys():
                                g_label_d[glabel]+=1
                            else:
                                g_label_d[glabel]=1
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
                                if glabel=='conj':
                                    cnt_root+=1

                                    #print(sent)

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

        print(cnt_root)

        return UAS, LAS, wrongs, wrong_conllu_list, g_label_d

    def forest_subord(self, mode1='wo_punct', mode2='vanilla', lang='en'):
        '''
        evaluator for forest rescoring approach
        '''
        correct = [{'UAS': [], 'LAS': []},{'UAS': [], 'LAS': []},{'UAS': [], 'LAS': []},{'UAS': [], 'LAS': []}]
        gold_d_list = self._make_gold_dict_subord(lang)

        wrongs = [[],[],[],[]]
        wrong_conllu_list = [[],[],[],[]]
        cnt_conllu=[0]*4
        cnt_line=[0]*4

        g_label_d = [dict(),dict(),dict(),dict()]
        cnt_root=0

        with open(self._pred_path, mode='r', encoding='utf-8') as p:
            pred_conllu_list = self._make_conllu_list(p.readlines())

            for i,pred_conllu in enumerate(pred_conllu_list):

                tmp_sent = []
                for line in pred_conllu:
                    tmp_sent.append(line[1].rstrip('\n'))

                sent = ' '.join(tmp_sent)
                if sent!=' ' and  sent!='':

                    i=0
                    for i,gold_d in enumerate(gold_d_list):
                        if sent in gold_d:
                            break

                    gold_conllu = gold_d_list[i][sent]
                    cnt_conllu[i]+=1

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
                        cnt_line[i]+=1

                        '''V-V
                        if gold[3][0]=='V' and len(pred_conllu)<=10 and gold[7] in {'advcl','ccomp','vmod','xcomp'}:
                            if gold_conllu[int(gold[6])-1][3][0]!='V':
                                continue
                        '''

                        '''V-nonV
                        if gold[3][0]=='V' and gold[7]=='advcl':
                            if gold_conllu[int(gold[6])-1][3][0]=='V':
                                continue
                        '''
                        ##advmod,aux,nsubj,mark,amod,dobj,acomp,pobj
                        if gold[3][0]=='V' and gold[7]=='ccomp':
                            if gold_conllu[int(gold[6])-1][3][0]=='V':
                                continue
                        #if gold[3]=='VERB' and gold[7] in {'advcl','advmod','vmod','xcomp'}:
                        #if 40<=len(pred_conllu) and gold[7] in {'advcl','ccomp','vmod','xcomp'}:
                        #if gold[3]=='VERB' and gold[7]=='ccomp':
                        #if gold[7]=='prep':

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
                                correct[i]['UAS'].append(0)
                                correct[i]['LAS'].append(0)
                                if glabel in g_label_d[i].keys():
                                    g_label_d[i][glabel]+=1
                                else:
                                    g_label_d[i][glabel]=1
                                ## replace with 1 if correct
                                ## check unlabeled attachment
                                if ghead == phead:
                                    correct[i]['UAS'][-1] = 1
                                ## check labeled attachment
                                if ghead == phead and glabel == plabel:
                                    correct[i]['LAS'][-1] = 1
                                ## if this conllu has an error
                                else:
                                    flag=True
                                    wrongs[i].append((ghead, phead, glabel, plabel))
                                    if glabel=='conj':
                                        cnt_root+=1

                    if flag==True:
                        wrong_conllu_list[i].append(pred_conllu)

        correct2 = [{k:np.array(v) for k, v in correct[i].items()} for i in range(4)]

        UAS = [(np.mean(correct2[i]['UAS']))*100 for i in range(4)]
        LAS = [(np.mean(correct2[i]['LAS']))*100 for i in range(4)]
        c1, c2 = [Counter(correct2[i]['UAS']) for i in range(4)], [Counter(correct2[i]['LAS']) for i in range(4)]

        ## total number of examples
        cnt=0
        for i in range(4):
            print('total conllus: '+str(cnt_conllu[i]))
            print('total scored lines: '+str(len(correct2[i]['UAS'])))
            print('total processed lines: '+str(cnt_line[i]))
            print('Head: '+str(c1[i]), 'Head&Label: '+str(c2[i]))
            cnt+=len(correct2[i]['UAS'])
        print('TOTAL scored lines: '+str(cnt))

        tmp = []
        a = [tmp.extend(list(correct2[i]['UAS'])) for i in range(0,4)]
        UAS_all = (np.mean(tmp)*100)
        tmp = []
        a = [tmp.extend(list(correct2[i]['LAS'])) for i in range(0,4)]
        LAS_all = (np.mean(tmp)*100)

        #c1, c2 = [Counter(correct2[i]['UAS']) for i in range(1,4)], [Counter(correct2[i]['LAS']) for i in range(1,4)]
        #print(cnt_root)

        return UAS, LAS, UAS_all, LAS_all, wrongs, wrong_conllu_list, g_label_d

    def forest_uas(self, mode1='wo_punct', mode2='vanilla'):
        '''
        evaluator for forest rescoring approach
        '''
        correct = {'UAS': []}
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
                    #print(line)
                    tmp_sent.append(line[1].rstrip('\n'))

                sent = ' '.join(tmp_sent)
                #print(tmp_sent)
                if sent!=' ' and  sent!='':
                    gold_conllu = gold_d[sent]
                    cnt_conllu+=1

                    flag=False
                    for pred,gold in zip(pred_conllu,gold_conllu):
                        cnt_line+=1

                        if mode2=='1best': ## modified
                            ## predicted
                            phead = pred[6]
                            ## correct answer
                            ghead = gold[6]
                        else:
                            ## predicted
                            phead = pred[8]
                            ## correct answer
                            ghead = gold[6]                           

                        ## check validity of each line
                        if mode1=='wo_punct':
                            valid = bool(len(pred) == 10 and gold[3] not in punct)
                        else:
                            valid = bool(len(pred) == 10)

                        ## give 1 for match / 0 for unmatch
                        if valid:
                            ## init with 0
                            correct['UAS'].append(0)
                            ## replace with 1 if correct
                            ## check unlabeled attachment
                            if ghead == phead:
                                correct['UAS'][-1] = 1
                            ## check labeled attachment
                            ## if this conllu has an error
                            else:
                                flag=True
                                wrongs.append((ghead, phead))

                    if flag==True:
                        wrong_conllu_list.append(pred_conllu)

        correct = {k:np.array(v) for k, v in correct.items()}

        UAS = (np.mean(correct['UAS']))*100
        c1 = Counter(correct['UAS'])

        ## total number of examples
        print('total conllus: '+str(cnt_conllu))
        print('total scored lines: '+str(len(correct['UAS'])))
        print('total processed lines: '+str(cnt_line))
        print('Head: '+str(c1))

        return UAS, wrongs, wrong_conllu_list

def analyze_deprel(wrongs,g_label_d):
    wrong_label = [wrong[2] for wrong in wrongs]
    #print(wrong_label)
    c = Counter(wrong_label)
    print(c)

    for label in c:
        cnt = g_label_d[label]
        rate=(cnt-c[label])/cnt
        if label in {'advcl','xcomp','ccomp','vmod'}:
            print(label,end=': ')
            print(rate)
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='file path for rescored cp predictions')
    parser.add_argument('--biaf_path', help='file path for 1best data')
    parser.add_argument('--gold_path', help='file path for gold data')
    parser.add_argument('--vanilla_path', help='file path for vanilla cp  prediction')
    parser.add_argument('--plist', nargs='+', type=str, help='a list of file paths to evaluate')
    parser.add_argument('--nbest', action='store_true', help='evaluate nbest list')
    parser.add_argument('--eisner_k', type=str, help='EisnerK')
    parser.add_argument('--K', type=str, help='K')
    parser.add_argument('--unlabel', action='store_true', default=False)
    parser.add_argument('--count_subord', action='store_true', default=False)

    parser.add_argument('--lang', type=str, default='en', help='en/ch')

    args = parser.parse_args()
    pred_path = args.pred_path
    biaf_path = args.biaf_path
    gold_path = args.gold_path
    plist = args.plist
    eval_nbest=args.nbest
    eisner_k = args.eisner_k
    K = args.K
    unlabel = args.unlabel
    count_subord = args.count_subord

    ## -----
    print('-----EVALUATION-----')
    if eval_nbest:
        print('---nbest---')
        evaluator = DepEvaluator(gold_path,pred_path)
        UAS, LAS = evaluator.nbest()
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
    
    else:
        vanilla_path = args.vanilla_path
        print('---biaffine 1best---')
        biaf_evaluator = DepEvaluator(gold_path, biaf_path)
        UAS, LAS, wrongs, _, g_label_d = biaf_evaluator.forest(mode1='wo_punct',mode2='1best')
        print('wo_punct')
        print('UAS: '+str(UAS), 'LAS: '+str(LAS))
        analyze_deprel(wrongs,g_label_d)

        if plist:
            pass
        else:
            if not unlabel:
                if count_subord:

                    print('---biaffine 1best---')
                    print(biaf_path)
                    wo_punct_evaluator = DepEvaluator(gold_path, biaf_path)
                    UAS, LAS, UAS_all, LAS_all, wrongs1, wrong_conlls1, g_label_d = wo_punct_evaluator.forest_subord(mode1='wo_punct',mode2='1best', lang=args.lang)
                    print('wo_punct')
                    for i in range(4):
                        print(i)
                        print('UAS: '+str(UAS[i]), 'LAS: '+str(LAS[i]))
                        analyze_deprel(wrongs1[i],g_label_d[i])
                    print(UAS_all, LAS_all)

                    print('---vanilla---')
                    print(vanilla_path)
                    vanilla_evaluator = DepEvaluator(gold_path, vanilla_path)
                    UAS, LAS, UAS_all, LAS_all, wrongs0, wrong_conlls0, g_label_d = vanilla_evaluator.forest_subord(mode1='wo_punct',mode2='vanilla', lang=args.lang)
                    print('wo_punct')
                    for i in range(4):
                        print(i)
                        print('UAS: '+str(UAS[i]), 'LAS: '+str(LAS[i]))
                        analyze_deprel(wrongs0[i],g_label_d[i])
                    print(UAS_all, LAS_all)

                    print('---rescored---')
                    print(pred_path)
                    wo_punct_evaluator = DepEvaluator(gold_path, pred_path)
                    UAS, LAS, UAS_all, LAS_all, wrongs1, wrong_conlls1, g_label_d = wo_punct_evaluator.forest_subord(mode1='wo_punct',mode2='rescored', lang=args.lang)
                    print('wo_punct')
                    for i in range(4):
                        print(i)
                        print('UAS: '+str(UAS[i]), 'LAS: '+str(LAS[i]))
                        analyze_deprel(wrongs1[i],g_label_d[i])
                    print(UAS_all, LAS_all)

                else:

                    print('---vanilla---')
                    '''
                    vanilla_name = 'vanilla_'+K+'.conllu'
                    dirname, basename = os.path.split(pred_path)
                    vanilla_path = os.path.join(dirname, vanilla_name)
                    '''
                    print(vanilla_path)
                    vanilla_evaluator = DepEvaluator(gold_path, vanilla_path)
                    UAS, LAS, wrongs0, wrong_conlls0, g_label_d = vanilla_evaluator.forest(mode1='wo_punct',mode2='vanilla')
                    print('wo_punct')
                    print('UAS: '+str(UAS), 'LAS: '+str(LAS))
                    analyze_deprel(wrongs0,g_label_d)

                    print('---rescored---')
                    print(pred_path)
                    wo_punct_evaluator = DepEvaluator(gold_path, pred_path)
                    UAS, LAS, wrongs1, wrong_conlls1, g_label_d = wo_punct_evaluator.forest(mode1='wo_punct',mode2='rescored')
                    print('wo_punct')
                    print('UAS: '+str(UAS), 'LAS: '+str(LAS))
                    analyze_deprel(wrongs1,g_label_d)
                    UAS, LAS, wrongs2, wrong_conlls2, g_label_d = wo_punct_evaluator.forest(mode1='w_punct',mode2='rescored')
                    print('w_punct')
                    print('UAS: '+str(UAS), 'LAS: '+str(LAS))    
                    analyze_deprel(wrongs2,g_label_d)
            
            else:
                print('UNLABEL')

                print('---vanilla---')
                '''
                vanilla_name = 'vanilla_'+K+'.conllu'
                dirname, basename = os.path.split(pred_path)
                vanilla_path = os.path.join(dirname, vanilla_name)
                '''
                print(vanilla_path)
                vanilla_evaluator = DepEvaluator(gold_path, vanilla_path, unlabel)
                UAS, wrongs0, wrong_conlls0 = vanilla_evaluator.forest_uas(mode1='wo_punct',mode2='vanilla')
                print('wo_punct')
                print('UAS: '+str(UAS))

                print('---rescored---')
                print(pred_path)
                wo_punct_evaluator = DepEvaluator(gold_path, pred_path, unlabel)
                UAS, wrongs1, wrong_conlls1 = wo_punct_evaluator.forest_uas(mode1='wo_punct',mode2='rescored')
                print('wo_punct')
                print('UAS: '+str(UAS))
                UAS, wrongs2, wrong_conlls2  = wo_punct_evaluator.forest_uas(mode1='w_punct',mode2='rescored')
                print('w_punct')
                print('UAS: '+str(UAS))    

