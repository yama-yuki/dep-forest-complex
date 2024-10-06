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

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from lib.conll import pkl_n_loader
import binarize2

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

def hypos2hyperedges(hypos):
    '''
    hyperedge = (Xspan, Aspan, Bspan, logp)
    hyperedge = (Xspan, Aspan, Bspan)
    '''
    hyperedges = set()
    for hypo in hypos:
        edges, logp = hypo
        actions, d_tails_labels = binarize2.binarize_actions(edges)
        print(edges)
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

def forest_format_hypo_unlabel(hyperedges,parse_probs,writer):
    
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
    ##id2prob = defaultdict()
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

        ## logp
        ##id2prob[X_id] = logp
    
    ##pprint(Xspan2ids)

    Xspans = topological_sort(sorted(hyperedge_ids))
    print('---')

    writer.writerow([str(len(Xspans))])

    error_id=set()
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
                    logp = np.log(parse_probs[tail,X]+1e-05)
                    ##logp = id2prob[X_id]
                    tails.add((Xspans.index(Aspan), Xspans.index(Bspan), logp))

            #line = "{0}\t{1}-{5} [{2}-{3}]\t{4}".format(i,Xspan[0],Xspan[1]+1,Xspan[2]+1,len(tails),Xspan[3])
            line = [i,str(Xspan[0])+'-'+str(Xspan[3])+' ['+str(Xspan[1]+1)+'-'+str(Xspan[2]+1)+']',len(tails)]
            print(line)
            writer.writerow(line)
        
            ##for tail in new_tails:
            for tail in tails:
                #line = "\t{0} {1} ||| 0={2}".format(tail[0], tail[1], tail[2])
                line = ['',str(tail[0])+' '+str(tail[1])+' ||| 0='+str(tail[2])]
                print(line)
                writer.writerow(line)

    writer.writerow([])

    return

def main_unlabel(paths):
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
            ##if cnt==76:
            hypos = forest['hypotheses']
            hyperedges = hypos2hyperedges(hypos)
            ##print(hyperedges)
            for he in hyperedges:
                pprint(he[0])

            ## sentence
            writer.writerow([str(cnt),str('ROOT '+' '.join(sent))])
            ## nodes and hyperedges
            forest_format_hypo_unlabel(hyperedges,parse_probs,writer)

            cnt+=1
    return

if __name__=='__main__':

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', help='file names from 1 to 10')
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

    main_unlabel(paths)
