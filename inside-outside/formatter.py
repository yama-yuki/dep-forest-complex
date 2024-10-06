'''
load hyperedges from .pkl
convert to the Forest format defined by Huang[08]
(to enable forest pruning)
'''

import csv
import sys
import os
from pprint import pprint
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from lib.conll import pkl_n_loader

import binarize

def topological_sort(hyperedge_ids):
    ## sort nodes based on its governing span length (ascending order)
    # hyperedge_id: 9_3_9_0_4_9_18 / X_A_B_a_c_b_lb / head_ltail_rtail_lmost_bound_rmost_deprel
    d = defaultdict()
    for hyperedge_id in hyperedge_ids:
        sp = list(map(int,hyperedge_id.split('_')))
        span_len = sp[5]-sp[3] 
        d[hyperedge_id] = span_len
    sorted_hyperedge_ids = [hyperedge_id for (hyperedge_id,_) in sorted(d.items(), key=lambda x: x[1])]

    Xspans = []
    for hyperedge_id in sorted_hyperedge_ids:
        X,_,_,a,_,b,l = list(map(int,hyperedge_id.split('_')))
        Xspan = (X,a,b,l)
        if Xspan not in Xspans:
            Xspans.append(Xspan) ## head,lmost,rmost

    print('Topologically Sorted Xspans: '+ str(Xspans))
    print(len(Xspans))

    return Xspans   

def add_terminal(Xspans):
    done = set()
    new = []
    for node in Xspans:
        if node not in done and node[2]-node[1]==2:
            left_terminal = (node[1]+1, node[1], node[1]+1)
            right_terminal = (node[2], node[1]+1, node[2])
            if left_terminal not in done:
                new.append(left_terminal)
                done.add(left_terminal)
            if right_terminal not in done:
                new.append(right_terminal)
                done.add(right_terminal)
        new.append(node)

    return new

def make_hyperedges(hes):
    hyperedges = set()
    for edges in hes:
        actions, d_tails_labels = binarize.binarize_actions(edges)
        print(actions)
        for i,action in enumerate(actions):
            X, tails, span_ranges, X_id = action
            tail = tails[1] if tails[0]==X else tails[0]
            Xspan = (X, span_ranges[0], span_ranges[2], d_tails_labels[(X,tail)], X_id)
            Aspan, Bspan = make_tail_spans(actions[i:], action, d_tails_labels)
            if Xspan and Aspan and Bspan:
                hyperedge = (Xspan, Aspan, Bspan)
                hyperedges.add(hyperedge)

    return hyperedges


def forest_format(hyperedges,parse_probs,rel_probs,writer):
    
    '''
    {((8, 6, 8, '8_7_8_6_7_8_5'),
    (7, 6, 7, '7_-2_-2_6_-2_7_-2'),
    (8, 7, 8, '8_-2_-2_7_-2_8_-2')),

    ((8, 6, 8, '8_7_8_6_7_8_7'),
    (7, 6, 7, '7_-2_-2_6_-2_7_-2'),
    (8, 7, 8, '8_-2_-2_7_-2_8_-2'))}
  '''

    hyperedge_ids = set()
    Xspan2ids = defaultdict(list)
    id2Xspan = defaultdict()
    Xspan2ABids = defaultdict()
    id2prob = defaultdict()
    for hyperedge in hyperedges:
        Xspan, Aspan, Bspan = hyperedge
        X,ax,bx,lx,X_id = Xspan
        A,aa,ba,la,A_id = Aspan
        B,ab,bb,lb,B_id = Bspan
        tail = A if X==B else B
        hyperedge_ids.add(Xspan[-1])
        hyperedge_ids.add(Aspan[-1])
        hyperedge_ids.add(Bspan[-1])
        id2Xspan[X_id] = (X,ax,bx,lx)
        id2Xspan[A_id] = (A,aa,ba,la)
        id2Xspan[B_id] = (B,ab,bb,lb)       
        Xspan2ids[Xspan[:-1]].append(X_id)
        Xspan2ABids[X_id] = [A_id, B_id]
    
    pprint(Xspan2ids)

    Xspans = topological_sort(sorted(hyperedge_ids))
    print('---')

    writer.writerow([str(len(Xspans))])

    for i,Xspan in enumerate(Xspans):
        X = Xspan[0]
        ## terminal
        if Xspan[3]==-2:
            #line = "{0}    {1} [{2}-{3}]    {4}".format(i,Xspan[0],Xspan[1]+1,Xspan[2]+1,0)
            line = [str(i),str(Xspan[0])+' ['+str(Xspan[1]+1)+'-'+str(Xspan[2]+1)+']',str(0)]
            print(line)
            writer.writerow(line)
            continue
        ## non-terminal    
        else:
            X_ids = Xspan2ids[Xspan]
            tails = set()
            for X_id in X_ids:
                lx = int(X_id.split('_')[-1])
                A_id, B_id = Xspan2ABids[X_id]
                Aspan, Bspan = id2Xspan[A_id], id2Xspan[B_id]
                if Aspan in Xspans and Bspan in Xspans:
                    A,B = Aspan[0], Bspan[0]
                    tail = A if X==B else B
                    parse_prob = parse_probs[tail,X]
                    rel_prob = rel_probs[tail,X,:][lx]
                    prob = parse_prob*rel_prob
                    tails.add((Xspans.index(Aspan), Xspans.index(Bspan), prob))

            #line = "{0}\t{1}-{5} [{2}-{3}]\t{4}".format(i,Xspan[0],Xspan[1]+1,Xspan[2]+1,len(tails),Xspan[3])
            line = [i,str(Xspan[0])+'-'+str(Xspan[3])+' ['+str(Xspan[1]+1)+'-'+str(Xspan[2]+1)+']',len(tails)]
            print(line)
            writer.writerow(line)
        
            for tail in tails:
                #line = "\t{0} {1} ||| 0={2}".format(tail[0], tail[1], tail[2])
                line = ['',str(tail[0])+' '+str(tail[1])+' ||| 0='+str(tail[2])]
                print(line)
                writer.writerow(line)

    writer.writerow([])

    return

def make_tail_spans(actions, action, d_tails_labels):
    X, tails, span_ranges, X_id = action
    A, B = tails
    a,c,b = span_ranges
    Aspan, Bspan = None, None
    ## make terminals
    if c-a == 1:## terminal
        A_id = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(A,-2,-2,a,-2,c,-2)
        Aspan = (A, a, c, -2, A_id)
    if b-c == 1:## terminal
        B_id = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(B,-2,-2,c,-2,b,-2)
        Bspan = (B, c, b, -2, B_id)
    if Aspan and Bspan:## both terminal
        return Aspan, Bspan
    
    ## make non-terminals
    if actions:
        for action in actions:
            N, N_tails, N_span_ranges, N_id = action

            if N==A and N_span_ranges[0]==a and N_span_ranges[2]==c:
                tail = N_tails[1] if N_tails[0]==N else N_tails[0]
                A_id = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(N,N_tails[0],N_tails[1],N_span_ranges[0],N_span_ranges[1],N_span_ranges[2],d_tails_labels[(N,tail)])
                Aspan = (A, a, c, d_tails_labels[(N,tail)], A_id)

            elif N==B and N_span_ranges[0]==c and N_span_ranges[2]==b:
                tail = N_tails[1] if N_tails[0]==N else N_tails[0]
                B_id = "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(N,N_tails[0],N_tails[1],N_span_ranges[0],N_span_ranges[1],N_span_ranges[2],d_tails_labels[(N,tail)])
                Bspan = (B, c, b, d_tails_labels[(N,tail)], B_id)

    if Aspan and Bspan:
        return Aspan, Bspan
    
    else:
        return None, None

def main(paths):
    ## out_path: .forest
    os.makedirs(paths['out_dir'], exist_ok=True)
    out_path = os.path.join(paths['out_dir'], paths['out_name'])    

    ## n: 1-10
    n=int(paths['n'])

    ## load from pkl 1-10 merged
    all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_n_loader(paths['forest_dir'],n)

    ## convert and write out
    cnt=0
    with open(out_path, mode='w', encoding='utf-8') as o:
        writer = csv.writer(o, delimiter='\t')
        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            hes = forest['hyperedges']
            hyperedges = make_hyperedges(hes)
            for he in hyperedges:
                pprint(he[0])

            writer.writerow([str(cnt),str('ROOT '+' '.join(sent))])
            
            forest_format(hyperedges,parse_probs,rel_probs,writer)

            cnt+=1
    return

def main_logp(path):
    ## out_path: .forest
    os.makedirs(paths['out_dir'], exist_ok=True)
    out_path = os.path.join(paths['out_dir'], paths['out_name'])    
    for_dir = paths['for_dir']
    forest_pkl = paths['for_pkl']

    ## n: 1-10
    n=int(paths['n'])

    ## load from pkl 1-10 merged
    all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_n_loader(paths['forest_dir'],n)


    return

if __name__=='__main__':

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', help='')
    argparser.add_argument('--forest_dir')
    argparser.add_argument('--forest_pkl')
    argparser.add_argument('--out_dir')
    argparser.add_argument('--out_name')
    args = argparser.parse_args()

    paths = {
        'n':args.n,
        'forest_dir':args.forest_dir,
        'forest_pkl':args.forest_pkl,
        'out_dir':args.out_dir,
        'out_name':args.out_name
    }

    main(paths)

'''
def topological_sort(hyperedge_ids):
    ## sort nodes based on its governing span length (ascending order)
    # hyperedge_id: 9_3_9_0_4_9_18 / X_A_B_a_c_b_lb / head_ltail_rtail_lmost_bound_rmost_deprel
    d = defaultdict()
    id2label = defaultdict()
    for (hyperedge_id,label) in hyperedge_ids:
        sp = list(map(int,hyperedge_id.split('_')))
        span_len = sp[5]-sp[3] 
        d[hyperedge_id] = span_len
        id2label[hyperedge_id] = label
    sorted_hyperedge_ids = [hyperedge_id for (hyperedge_id,_) in sorted(d.items(), key=lambda x: x[1])]

    Xspans = []
    for hyperedge_id in sorted_hyperedge_ids:
        X,_,_,a,_,b,_ = list(map(int,hyperedge_id.split('_')))
        label = id2label[hyperedge_id]
        Xspan = (X,a,b,label)
        if Xspan not in Xspans:
            Xspans.append(Xspan) ## head,lmost,rmost

    print('Topologically Sorted Xspans: '+ str(Xspans))

    return Xspans

def forest_format(hes, o):

    ## organize as dict
    node2he = defaultdict(list)
    hyperedge_ids = []
    for he in hes:
        head, head_span = he['head'], he['head_span']
        hyperedge_ids.append((he['name'],he['label']))
        ## node: (head , left, right, label)
        node = (head, head_span[0], head_span[1], he['label'])
        node2he[node].append(he)
    nodes = topological_sort(hyperedge_ids)
    print('---')
    nodes = add_terminal(nodes)
    print(nodes)

    for i,node in enumerate(nodes):

        hes = node2he[node]
        tail_nodes = []
        for he in hes:
            head, left_tail, right_tail = he['head'], he['left_tail'], he['right_tail']
            head_range, left_range, right_range = he['head_span'], he['left_span'], he['right_span']
            label, prob = he['label'], he['prob']
            left_span = (left_tail, left_range[0], left_range[1], label)
            right_span = (right_tail, right_range[0], right_range[1], label)
            tail_nodes.append([nodes.index(left_span), nodes.index(right_span), prob])

        if len(node)>3: ##non-terminal
            line = "{0}    {1}-{5} [{2}-{3}]    {4}".format(i,node[0],node[1]+1,node[2]+1,len(hes),node[3])
        else: ## terminal
            line = "{0}    {1} [{2}-{3}]    {4}".format(i,node[0],node[1]+1,node[2]+1,len(hes))
        print('\n')
        print(line)
        o.write(line+'\n')
        for tail in tail_nodes:
            line = "    {0} {1}  ||| 0={2}".format(tail[0], tail[1], tail[2])
            print(line)
            o.write(line+'\n')

    o.write('\n')

    return

def forest_format2(hes, o):
    ## organize as dict
    Xspan2he = defaultdict(list)
    span_ids = []
    terminal = []
    for he in hes:
        Xspan = he['Xspan']
        span_ids.append((he['X_id'],Xspan[3]))
        Xspan2he[Xspan].append(he)

        if he['Aspan'][3]==None:
            span_ids.append((he['A_id'],None))
            terminal.append(he['Aspan'])
        if he['Bspan'][3]==None:
            span_ids.append((he['B_id'],None))
            terminal.append(he['Bspan'])

    Xspans = topological_sort(span_ids)
    print('---')

    for i,Xspan in enumerate(Xspans):

        ## terminal
        if Xspan[3]==None:
            line = "{0}    {1} [{2}-{3}]    {4}".format(i,Xspan[0],Xspan[1]+1,Xspan[2]+1,0)
            print(line)
            o.write(line+'\n')
            continue
        ## non-terminal    
        else:
            hes = Xspan2he[Xspan]
            tails = []
            for he in hes:
                Aspan, Bspan = he['Aspan'], he['Bspan']
                parse_prob = he['parse_prob']
                if Aspan in Xspans and Bspan in Xspans:
                    tails.append([Xspans.index(Aspan), Xspans.index(Bspan), parse_prob])
            line = "{0}    {1}-{5} [{2}-{3}]    {4}".format(i,Xspan[0],Xspan[1]+1,Xspan[2]+1,len(tails),Xspan[3])
            print(line)
            o.write(line+'\n')
        
            for tail in tails:
                line = "    {0} {1}  ||| 0={2}".format(tail[0], tail[1], tail[2])
                print(line)
                o.write(line+'\n')

    o.write('\n')

    #pprint(Xspans)

    return

def main(paths):
    ## out_path: .forest
    os.makedirs(paths['out_dir'], exist_ok=True)
    out_path = os.path.join(paths['out_dir'], paths['out_name'])

    ## load from pkl
    all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_loader(paths['forest_dir'])
    
    ## convert and write out
    with open(out_path, mode='w', encoding='utf-8') as o:
        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            hes = forest['hyperedges']
            forest_format(hes, o)
            
            break

    return

def main2(paths):
    ## out_path: .forest
    os.makedirs(paths['out_dir'], exist_ok=True)
    out_path = os.path.join(paths['out_dir'], paths['out_name'])    

    ## load from pkl
    all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_loader(paths['forest_dir'])

    ## convert and write out
    with open(out_path, mode='w', encoding='utf-8') as o:
        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            hes = forest['hyperedges']
            forest_format2(hes, o)
            break

    return

'''
'''
{
'hyperedges': [{'name': '0_0_1_-1_0_1_1', 'head': 0, 'left_tail': 0, 'right_tail': 1, 'head_span': [-1, 1], 'left_span': [-1, 0], 'right_span': [0, 1], 'label_id': 1, 'label': 'root', 'prob': '0.010450535'}, {'name': '2_1_2_0_1_2_43', 'head': 2, 'left_tail': 1, 'right_tail': 2, 'head_span': [0, 2], 'left_span': [0, 1], 'right_span': [1, 2], 'label_id': 43, 'label': 'discourse', 'prob': '7.9171275e-07'}],
'nodes':['0_0_1_-1_0_1_1', '2_1_2_0_1_2_43', '2_1_2_0_1_2_12', '2_1_2_0_1_2_26', '2_1_2_0_1_2_30', '2_1_2_0_1_2_19', '2_1_2_0_1_2_14'],
'node_ids':[0, 1, 2, 3, 4, 5, 6, 7, 8]
}
'''

