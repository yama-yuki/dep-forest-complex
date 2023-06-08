import csv
import os
from socket import TCP_FASTOPEN
from tqdm import tqdm
from collections import Counter
from stanza.utils.conll import CoNLL

def load_data(data_path):
    snt_list, pred_i_list, head_i_list = [], [], []
    with open(data_path, mode='r', encoding='utf-8') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            snt,pred_i,head_i = row[:3] ## count id from 0
            snt_list.append(snt)
            pred_i_list.append(pred_i)
            head_i_list.append(head_i)
    return snt_list, pred_i_list, head_i_list

if __name__ == '__main__':
    mode = 'ewt'

    results = []
    for num in tqdm(range(3)):
        num=num+1
        parsed_path = os.path.join('../data/baseline/parse', 'stanza_'+mode+'_'+str(num)+'.conllu')
        data_path = '../data/baseline/'+mode+'/ud'+str(num)+'.tsv'

        _, pred_i_list, head_i_list = load_data(data_path)
        doc = CoNLL.conll2doc(parsed_path)

        #print(data_path)
        #print(len(pred_i_list), len(head_i_list))
        #print(len(doc.sentences))

        tf_list = []
        for i,sentence in enumerate(doc.sentences):
            pred_i = int(pred_i_list[i])#+1
            head_i = int(head_i_list[i])#+1
            #print(pred_i, head_i)
            for word in sentence.words:
                if word.id == pred_i:
                    #print(word.head)
                    tf_list.append(word.head == head_i)
        
        c = Counter(tf_list)
        print(c)
        f1 = (c[True]/(c[True]+c[False]))*100

        results.append([str(num),str(f1),len(doc.sentences)])

        with open('../pred/stanza/'+mode+'_'+str(num)+'.csv', mode='w', encoding='utf8') as o:
            tf10_list = [int(tf) for tf in tf_list]
            print(tf10_list)
            writer = csv.writer(o)
            writer.writerow(tf10_list)
    
    for result in results:
        print(result)
