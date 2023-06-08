import argparse
import sys
import numpy as np

def make_conllu_list(lines):

    conllu_list = []
    tmp = []
    for line in lines:
        if line=='' or line[0]=='#':
            conllu_list.append(tmp)
            tmp = []
            continue
        sp = line.rstrip('\n').split('\t')
        tmp.append(sp)
    conllu_list.append(tmp)

    return conllu_list

def evaluate_from_conllu_list(conllu_list):
    punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', help='file path for predictions')
    parser.add_argument('--data_path', help='file path for test data')
    args = parser.parse_args()
    pred_path = args.pred_path
    data_path = args.data_path

    with open(pred_path, mode='r', encoding='utf-8') as p:
        with open(data_path, mode='r', encoding='utf-8') as t:
            ##['1', 'We', '_', 'PRON', 'PRP', '_', '2', 'nsubj', '2', 'nsubj']

            test_conllu_list = make_conllu_list(t.readlines())
            print(test_conllu_list)

            sys.exit()
            pred_conllu_list = make_conllu_list(p.readlines())
            print(pred_conllu_list)

    for i,pred_conllu,test_conllu in enumerate(zip(pred_conllu_list,test_conllu_list)):

        for pred,test in zip(pred_conllu,test_conllu):
            phead,plabel = pred[6:8]
            thead,tlabel = test[6:8]



