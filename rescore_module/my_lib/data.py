'''
script for creating dataset
deptree (.conllu) -> squad (.json)

config file: 'data.cfg'
'''

import csv,json,pickle
import glob
import random
import os,sys
from collections import defaultdict
from conllu import parse_incr, parse_tree_incr
from tqdm import tqdm

import stanza
from stanza.utils.conll import CoNLL

def tree_linearize(in_path, search, with_root):
    """linearize conllu format
    Args:
        in_path (.conllu): dependency tree
    Returns:
        results: a list of linearized trees
    """

    lintrees_2, lintrees_3, answer_ids, answer_toks, sentences = [], [], [], [], []
    with open(in_path, 'r', encoding='utf-8') as f:
        #for root in parse_tree_incr(f):
        for root in tqdm(parse_incr(f)):#yields toklist
            ids = check_subordination_with_root(root, with_root) ##head, tail
            if ids:
                #print(ids)
                sent_list = [tok['form'] for tok in root]
                sentence = ' '.join(sent_list)
                #print(sentence)
                answer_ids.append(ids[0])
                answer_toks.append(sent_list[ids[0]-1])
                sentences.append(sentence)

                if search=='dfs':
                    root = root.to_tree()
                    linearized_tree = DFS(root,[],ids[1],O=2)
                    lintrees_2.append(linearized_tree)
                    linearized_tree = DFS(root,[],ids[1],O=3)
                    lintrees_3.append(linearized_tree)
                    #print(linearized_tree)
                
                elif search=='bfs':
                    #print(ids[1])
                    start_tok = root[ids[1]-1]
                    tree = root.to_tree()
                    #print(tree)
                    lintree2, lintree3 = BFS(tree,ids[1],O=3)
                    #sys.exit()
                    lintrees_2.append(lintree2)
                    lintrees_3.append(lintree3)
                
                elif search=='word2word':
                    lintrees_2.append('None')
                    lintrees_3.append('None')                    

    return [lintrees_2, lintrees_3, answer_ids, answer_toks, sentences]

def ud_linearize(in_path, search, with_root, s):
    """
    Args:
        in_path (.conllu): dependency tree
        search: dfs/bfs
    Returns:
        results: a list of linearized trees
    """

    lintrees_2, lintrees_3, answer_ids, answer_toks, sentences = [], [], [], [], []
    with open(in_path, 'r', encoding='utf-8') as f:
        #for root in parse_tree_incr(f):
        for root in tqdm(parse_incr(f)):#yields toklist
            sconj_num = count_sconj(root)
            if sconj_num==s:
                ids = check_subordination_with_root(root, with_root)
                if ids:
                    #print(ids)
                    sent_list = [tok['form'] for tok in root]
                    sentence = ' '.join(sent_list)
                    #print(sentence)
                    answer_ids.append(ids[0])
                    answer_toks.append(sent_list[ids[0]-1])
                    sentences.append(sentence)

                    if search=='dfs':
                        root = root.to_tree()
                        linearized_tree = DFS(root,[],ids[1],O=2)
                        lintrees_2.append(linearized_tree)
                        linearized_tree = DFS(root,[],ids[1],O=3)
                        lintrees_3.append(linearized_tree)
                        #print(linearized_tree)
                    
                    elif search=='bfs':
                        #print(ids[1])
                        start_tok = root[ids[1]-1]
                        tree = root.to_tree()
                        #print(tree)
                        lintree2, lintree3 = BFS(tree,ids[1],O=3)
                        #sys.exit()
                        lintrees_2.append(lintree2)
                        lintrees_3.append(lintree3)

    return [lintrees_2, lintrees_3, answer_ids, answer_toks, sentences]

def pickle_out(out_path, extracted_data):
    lintrees_2, lintrees_3, answer_ids, answer_toks, sentences = extracted_data

    if len(lintrees_2) == len(lintrees_3) == len(answer_ids) == len(answer_toks) == len(sentences):
        d = defaultdict(list)
        d['lintree_2'] = lintrees_2
        d['lintree_3'] = lintrees_3
        d['answer_id'] = answer_ids
        d['answer_tok'] = answer_toks
        d['sentence'] = sentences
        with open(out_path,'wb') as o:
            pickle.dump(d, o)

    else:
        print(len(sentences))
        print(len(lintrees_2))
        print(len(lintrees_3))
        print(len(answer_ids))
        print(len(answer_toks))
        sys.exit('ERROR: inconsistent list lengths')

def pickle_read(pkl_path):
    with open(pkl_path,'rb') as f:
        d = pickle.load(f)
    return d

def DFS(tree,N,start_id,O=1,n=0,flag=False):
    """for n-th order
    Arguments:
        tree (TokTree): top node of a TokTree
        N (list): current state of linearization
        start_id (int): id of the head node
        O (int): 1st/2nd/3rd-order
        n (int): current tree depth
        flag (bool): True/False

    Returns:
        N:
    """    
    if n==0 and tree.token['id']==start_id:
        N.append(tree.token['form'])
        flag=True
        n+=1

    elif flag==True and O>=n:
        N.append(tree.token['form'])

    nodes = tree.children
    if nodes:
        if flag==True:
            n+=1
        if flag==True and O>=n:
            N.append('(')
        for node in nodes:
            DFS(node,N,start_id,O,n,flag)
        if flag==True and O>=n:
            N.append(')')
    return N

def BFS(tree, start_id, O):

    def find_start(tree, start_id):
        if tree.children:
            for tok in tree.children:
                if tok.token['id']==start_id:
                    return tok
                else:
                    tok = find_start(tok, start_id)
                if tok:
                    return tok

    tok = find_start(tree, start_id)

    ## O=1
    N = [tok.token['form']]
    N1 = N

    ## O=2
    children = [child for child in tok.children]
    if children:
        second = [child.token['form'] for child in children]
        second_str = ' ' + ' '.join(second) + ' '
        N.append('(')
        N.extend(second)
        N.append(')')
        N2 = N
    
    else:
        return N1, N1

    ## O=3
    grand_children = [grand_child for child in children for grand_child in child.children]
    if grand_children:
        third = [grand_child.token['form'] for grand_child in grand_children]
        third_str = ' ' + ' '.join(third) + ' '
        N.append('(')
        N.extend(third)
        N.append(')')
        N3 = N

    else:
        return N2, N2

    return N2, N3

def check_subordination_with_root(sentence, with_root):
    '''find subordinating structure
    Argument:
        sentence: TokList

    Returns:
        Tuple(head_id (int), child_id (int))
    '''
    for tok in sentence:
        if tok['deprel']=='advcl'and tok['upos'][0]=='V':
            head_id = tok['head']
            if head_id:
                try:
                    if with_root:
                        if sentence[head_id-1]['upos'][0] == 'V' or head_id==0:
                            child_id = tok['id']
                            return (head_id, child_id)
                    else:
                        if sentence[head_id-1]['upos'][0] == 'V':
                            child_id = tok['id']                            
                            return (head_id, child_id)
                except:
                    print('ERROR at '+str(head_id))

    return

def count_sconj(sentence):
    count=0
    sconj = []

    for i,tok in enumerate(sentence):
        if tok['upos'] == 'SCONJ':
            sconj.append(tok['form'])
            count+=1

    return count

def rewrite_conll(in_path, out_path):
    '''
    normalize spaces in a conll file to avoid error when reading with stanza and etc.
    '''
    sentences = []
    with open(in_path, "r", encoding="utf-8") as f:
        with open(out_path, 'w', encoding="utf-8") as o:
            for sentence in tqdm(parse_incr(f)):
                sentences.append(sentence)
            o.writelines([sentence.serialize() + "\n" for sentence in tqdm(sentences)])
    return

def parse_data(mode, with_root, in_path, out_path, s):

    def parse(snt_list, s):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)

        out_path = os.path.join('../data/baseline/parse', 'stanza_'+mode+'_'+str(s)+'.conllu')
        doc = nlp(snt_list)
        CoNLL.write_doc2conll(doc, out_path)

    snt_list = []
    rows = []
    with open(in_path, 'r', encoding='utf-8') as f:
        #for root in parse_tree_incr(f):
        for root in tqdm(parse_incr(f)):#yields toklist
            temp = []
            sconj_num = count_sconj(root)
            if sconj_num==s:
                ids = check_subordination_with_root(root, with_root) ##head,tail
                if ids:
                    sent_list = [tok['form'] for tok in root]
                    sentence = ' '.join(sent_list)
                    snt_list.append(sent_list)
                    temp = [sentence, ids[1], ids[0], sent_list[ids[1]-1], sent_list[ids[0]-1]]
                    rows.append(temp)

    with open(out_path, mode='w', encoding='utf-8') as o:
        writer = csv.writer(o, delimiter='\t')
        writer.writerows(rows)

    parse(snt_list, s)

def extract(mode, search, data_type, pkl_dir, with_root, conll_data_dir='None'):

    if mode == 'debug':
        test_path = '../data/debug/test.conllu'
        out_path = '../data/debug/test.out'
        lintrees_2, lintrees_3, answer_ids, answer_toks, sentences = tree_linearize(test_path, search, with_root)
        print([' '.join(lintree_3) for lintree_3 in lintrees_3])
        #pickle_out(out_path, lintrees_2, lintrees_3, answer_ids, answer_toks, sentences)
        #with open(out_path,'rb') as f:
            #d = pickle.load(f)

        return

    elif mode == 'wiki':
        ## alpha = ['A','B','C','D','E','F','G']
        alpha = [os.path.splitext(os.path.basename(file_))[0] for file_ in glob.glob(os.path.join(conll_data_dir,'*.conllu'))]
        for a in tqdm(alpha):
            in_path = os.path.join(conll_data_dir,a+'.conllu')
            if search == 'bfs':
                out_path = os.path.join(pkl_dir,a+'_bfs.pkl')
            else:
                out_path = os.path.join(pkl_dir,a+'.pkl')

            ## convert .conllu to linearized trees
            pickle_out(out_path, tree_linearize(in_path, search, with_root))
        
        return alpha

    elif mode=='ud':
        subord = [1, 2, 3, 4, 5]
        for s in tqdm(subord):
            in_path = os.path.join(conll_data_dir,data_type+'.conllu')
            if search == 'bfs':
                out_path = os.path.join(pkl_dir,'ud'+str(s)+'_bfs.pkl')
            elif search == 'dfs':
                out_path = os.path.join(pkl_dir,'ud'+str(s)+'.pkl')
            elif search == 'stanza':
                baseline_dir = '../data/baseline'
                out_path = os.path.join(baseline_dir,'ud','ud'+str(s)+'.tsv')
                parse_data(mode, with_root, in_path, out_path, s)
                continue
            else:
                sys.exit()
            pickle_out(out_path, ud_linearize(in_path, search, with_root, s))
        
        return

    elif mode == 'ud_pattern':
        in_path = '../data/ud/udbank.conllu'
        subord_patterns = {'2': ['00', '01'], '3': ['000', '001', '010', '011']}
        for k,v_list in subord_patterns.items():
            for v in v_list:
                out_path = '../data/ud/ud'+str(k)+str(v)+'.pkl'
                print(k, v)

        return

def pickle_merge(search, pkl_dir, alpha):
    '''
    merge all .pkl files
    '''
    results = defaultdict(list)

    print('merging .pkl')

    for a in tqdm(alpha):
        if search=='bfs':
            pkl_path = os.path.join(pkl_dir,a+'_bfs.pkl')

        else:
            pkl_path = os.path.join(pkl_dir,a+'.pkl')

        with open(pkl_path, mode='rb') as f:
            dic = pickle.load(f)
            sentences = dic['sentence']
            lintrees2, lintrees3 = dic['lintree_2'], dic['lintree_3']
            answer_toks, answer_ids = dic['answer_tok'], dic['answer_id']

            results['sentence'].extend(sentences)
            results['lintree_2'].extend(lintrees2)
            results['lintree_3'].extend(lintrees3)
            results['answer_tok'].extend(answer_toks)
            results['answer_id'].extend(answer_ids)
            print(len(sentences))
    
    return results

def to_squad(mode, search, pkl_dir, out_dir, alpha=None):
    ## defaultdict(<class 'list'>, {'lintree': [['becoming', '(', ',', 'Priest', '(', 'the', 'first', 'High', 'Israelites', ')', ')'], ['having', '(', 'bonds', '(', 'single', ')', 'atoms', '(', 'to', 'three', 'other', ')', ')']], 'answer_id': [2, 9], 'answer_tok': ['represented', 'saturated'], 'sentence': ['He represented the priestly functions of his tribe , becoming the first High Priest of the Israelites .', 'In particular , this carbon center should be saturated , having single bonds to three other atoms .']})

    ## {"data": [{"title": "None", "paragraphs": 
    ## [{ "context": <context>, "qas": [
    ## {"answers": [{"answer_start": <id>, "text": <ans>}], "question": <question>, "id": <id>},

    if mode=='wiki':
        dic = pickle_merge(search, pkl_dir, alpha)

        sentences=dic['sentence']
        lintrees2,lintrees3=dic['lintree_2'],dic['lintree_3']
        answer_toks,answer_ids=dic['answer_tok'],dic['answer_id']
        print(len(sentences))

        orders = [1, 2, 3]
        print('create json')
        for order in tqdm(orders):
            new_data = {"data": [{"title": "None", "paragraphs":[]}]}

            for i in tqdm(range(len(sentences))):
                entry = {"context":"","qas":[]}

                entry["context"] = sentences[i]
                qas = []
                d = {"answers": [{"answer_start": "", "text": ""}], "question": "", "id": ""}
                d["answers"][0]["text"] = answer_toks[i] #head_word
                d["answers"][0]["answer_start"] = answer_ids[i]
                snt = sentences[i].split(' ')
                d["answers"][0]["answer_start"] = len(' '.join(snt[:answer_ids[i]-1]))+1 #contextを0からstrのcount

                if order==1:
                    d["question"] = lintrees2[i][0]
                elif order==2:
                    d["question"] = ' '.join(lintrees2[i])
                elif order==3:
                    d["question"] = ' '.join(lintrees3[i])

                d["id"] = str(i+1)
                qas.append(d)

                entry["qas"] = qas
                new_data["data"][0]["paragraphs"].append(entry)

            #print(new_data)
            if search == 'bfs':
                out_path = os.path.join(out_dir,str(order)+'_bfs.json')
            else:
                out_path = os.path.join(out_dir,str(order)+'.json')
            with open(out_path, 'w') as o:
                json.dump(new_data, o)

    elif mode in {'ewt', 'gum', 'partut'}:
        subord = [1, 2, 3, 4, 5]
        for s in tqdm(subord):
            with open(os.path.join(pkl_dir,'ud'+str(s)+'.pkl'), mode='rb') as f:
                dic = pickle.load(f)

            sentences=dic['sentence']
            lintrees2,lintrees3=dic['lintree_2'],dic['lintree_3']
            answer_toks,answer_ids=dic['answer_tok'],dic['answer_id']
            print(len(sentences))

            orders = [1, 2, 3]
            print('create json')
            for order in tqdm(orders):
                new_data = {"data": [{"title": "None", "paragraphs":[]}]}

                for i in tqdm(range(len(sentences))):
                    entry = {"context":"","qas":[]}

                    entry["context"] = sentences[i]
                    qas = []
                    d = {"answers": [{"answer_start": "", "text": ""}], "question": "", "id": ""}
                    d["answers"][0]["text"] = answer_toks[i] #head_word
                    d["answers"][0]["answer_start"] = answer_ids[i]
                    snt = sentences[i].split(' ')
                    d["answers"][0]["answer_start"] = len(' '.join(snt[:answer_ids[i]-1]))+1 #contextを0からstrのcount

                    if order==1:
                        d["question"] = lintrees2[i][0]
                    elif order==2:
                        d["question"] = ' '.join(lintrees2[i])
                    elif order==3:
                        d["question"] = ' '.join(lintrees3[i])

                    d["id"] = str(i+1)
                    qas.append(d)

                    entry["qas"] = qas
                    new_data["data"][0]["paragraphs"].append(entry)
                    if search == 'bfs':
                        out_path = os.path.join(out_dir,'ud'+str(s)+'_'+str(order)+'_bfs.json')
                    else:
                        out_path = os.path.join(out_dir,'ud'+str(s)+'_'+str(order)+'.json')
                    
                    with open(out_path, 'w') as o:
                        json.dump(new_data, o)

def split_dev_test(search, order, data_dir):
    '''
    split squad data into train&dev&test
    '''
    random.seed(0)

    dev_data = {"data": [{"title": "None", "paragraphs":[]}]}
    test_data = {"data": [{"title": "None", "paragraphs":[]}]}

    ## read squad data
    if search == 'bfs':
        with open(os.path.join(data_dir,str(order)+'_bfs.json')) as f:
            data = json.load(f)
            print('Total: '+str(len(data["data"][0]["paragraphs"])))
    else:
        with open(os.path.join(data_dir,str(order)+'.json')) as f:
            data = json.load(f)
            print('Total: '+str(len(data["data"][0]["paragraphs"])))

    ## split dev_data
    num_list = random.sample(range(len(data["data"][0]["paragraphs"])-10000), k=10000)
    for num in num_list:
        dev_data["data"][0]["paragraphs"].append(data["data"][0]["paragraphs"].pop(num))

    ## split test_data
    num_list = random.sample(range(len(data["data"][0]["paragraphs"])-10000), k=10000)
    for num in num_list:
        test_data["data"][0]["paragraphs"].append(data["data"][0]["paragraphs"].pop(num))

    ## the rest is train_data
    train_data = data

    print('Train: '+str(len(train_data["data"][0]["paragraphs"])))
    print('Dev: '+str(len(dev_data["data"][0]["paragraphs"])))
    print('Test: '+str(len(test_data["data"][0]["paragraphs"])))
    
    data_list = [train_data, dev_data, test_data]
    out_name = ['train'+str(order), 'dev'+str(order), 'test'+str(order)]

    ## write out split squad data
    for i in range(len(data_list)):
        if search=='bfs':    
            out_path = os.path.join(data_dir,out_name[i]+'_bfs.json')
        else:
            out_path = os.path.join(data_dir,out_name[i]+'.json')

        with open(out_path, mode='w') as o:
                json.dump(data_list[i],o)

if __name__ == '__main__':

    import configparser
    cfg = configparser.RawConfigParser()
    cfg.read('data.cfg')

    if cfg['Mode']['mode']=='debug':
        extract('debug', cfg['Options']['search'], cfg['Options']['data_type'], cfg['OS']['pkl_dir'], cfg['Options']['with_root'])
    
    elif cfg['Mode']['mode']=='ud':
        ## step1: extract complex sents from udbank and save as .pkl
        extract('ud', cfg['Options']['search'], cfg['Options']['data_type'], cfg['OS']['pkl_dir'], cfg['Options']['with_root'], conll_data_dir=cfg['OS']['ud_data'])

        ## step2: convert .pkl into squad_style .json
        if cfg['Options']['search']!='stanza':
            to_squad('ud', cfg['Options']['search'], cfg['OS']['out_dir'], cfg['OS']['pkl_dir'])
    
    elif cfg['Mode']['mode']=='wiki':
        ## step1: extract complex sents from wikipedia and save as .pkl
        alpha = extract('wiki', cfg['Options']['search'], cfg['Options']['data_type'], cfg['OS']['pkl_dir'], cfg['Options']['with_root'], conll_data_dir=cfg['OS']['wiki_data'])
    
        ## step2: convert .pkl into squad_style .json
        to_squad('wiki', cfg['Options']['search'], cfg['OS']['out_dir'], cfg['OS']['pkl_dir'], alpha)
        
        ## step3: split into train&dev&test set
        orders = list(range(1,int(cfg['Options']['order'])+1)) #orders = [1,2,3]
        for order in tqdm(orders):
            split_dev_test(cfg['Options']['search'], cfg['Options']['order'], cfg['OS']['out_dir'])

