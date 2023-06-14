'''
Includes:
(A) Forest Reader: take out nodes from forest and sort them for cube pruning
(B) Cube Pruning Algorithm: the part where search/rescore happens
(C) Rescoring Function: rescoring span combinations inside cubes
'''

import argparse
import configparser
import os, sys
import json
import heapq
import numpy as np
import pickle as pkl
from collections import Counter, defaultdict
from pprint import pprint
from tqdm import tqdm

from rescore_module.my_lib.rescore_main import RescoreModel
from lib.conll import final_1best, to_conllu, pkl_loader

## derivation D
class HypoD:
    ## tree hypotheses
    def __init__(self, acclogp, node, depedges_l, depedges, node_names, c, num_roots):
        self.acclogp = acclogp
        self.node = int(node)
        self.depedges_l = depedges_l
        self.depedges = depedges
        self.node_names = node_names
        self.c = c
        self.num_roots = num_roots

    def to_list(self):
        print(str(self.node))
        print(list(self.depedges_l))
        print(list(self.depedges))
        print(str(self.acclogp))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (A) Forest Reader
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def id2hyperedge(forest):
    '''
    forest_d[node_name]: hyperedge
    '''
    forest_d = defaultdict()
    for he in forest['hyperedges']:
        forest_d[he['name']] = he

    return forest_d

def Xspan2hyperedgeid(forest_d, Xspans):
    '''
    In:
    Xspans: list
        Xspan = (X, a, b)

    Out:
    Xspan_forest_d: dict
        Xspan_forest_d[Xspan] = hyperedge_id
    '''
    Xspan_forest_d = defaultdict(list)
    for Xspan in Xspans:
        X, a, b = Xspan
        for he_name in forest_d:
            sp = list(map(int,he_name.split('_')))
            if sp[0]==X and sp[3]==a and sp[5]==b:
                Xspan_forest_d[Xspan].append(he_name)
            
    return Xspan_forest_d

def load_forest(forest):
    '''
    span variables:
    X: head node
    A: left tail
    B: right tail
    a: left most id
    c: boundary id
    b: right most id
    lb: deprel id

    hyperedge_id: X_A_B_a_c_b_lb
    Xspan: X, a, b
    forest_d: hyperedge_id -> hyperedge
    Xspan_forest_d: (X, a, b) -> list(hyperedge_ids)
    '''

    ## a list of hyperedge_ids
    hyperedge_ids = forest['nodes']
    ## hyperedge_id->hyperedge
    forest_d = id2hyperedge(forest)
    ## sorting hyperedges based on span_len
    Xspans = topological_sort(hyperedge_ids)
    ## node_span_range->[hyperedges]
    Xspan_forest_d = Xspan2hyperedgeid(forest_d, Xspans)

    return forest_d, Xspans, Xspan_forest_d

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
        X,_,_,a,_,b,_ = list(map(int,hyperedge_id.split('_')))
        Xspan = (X,a,b)
        if Xspan not in Xspans:
            Xspans.append(Xspan) ## head,lmost,rmost

    print('Topologically Sorted Xspans: '+ str(Xspans))

    return Xspans

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (B) Cube Pruning Algorithm
## This algorithm searches Kbest derivations of a Xspan which is a triplet of X(head node), a(leftmost governing span boundary of X), and b(rightmost governing span boundary of X)
## Then returns derivation[(0,-1,length-1)][0], the resulting 1best dependency tree with a root node governing leftmost to rightmost
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, sent, tags, out_path, K):
    length = len(forest['node_ids']) # hyperedge_ids
    print('sentence length: '+str(length-1))
    #print('with root: '+str(length))
    print('total nodes: '+str(len(forest['nodes'])))

    ## get sorted list of nodes from a parsed forest
    forest_d, Xspans, Xspan_forest_d = load_forest(forest)

    ## initializing chart of derivations
    ## key:Xspan(X,a,b) # (1,0,1), ... (9,0,9), (1,1,2)
    terminals = set()
    derivations = defaultdict(list)
    for X in range(0,length+1): ## X loop 0-9
        for A in range(-1,length): ## A loop 0-9, will be used as 'a'
            for B in range(0,length+1): ## B loop 0-9, will be used as 'b'
                ## check if it is a terminal edge (i.e. bottom/terminal)
                if B-A==1 and A<=X<=B:
                    terminals.add((X,A,B))
                derivations[(X,A,B)].append((0.0, 0, HypoD(0.0, X, set(), set(), set(), None, 0)))
                ## node, acclogp, depedges_l, depedges, c, num_roots
                ## Aspan, Bspan = (A,a,c), (B,c,b)
    print('Inialized D: '+str(derivations))

    ## main loop
    '''Bottom-up combinatory operation of span flagments
    e.g., Aspan(0,-1,1) + Bspan(2,1,9) -> Xspan(0,-1,9)
    
    Xspans:
    [(0, -1, 1), (2, 0, 2), (1, 0, 2), (3, 1, 3), (2, 1, 3), (4, 2, 4), (3, 2, 4), (5, 3, 5), (4, 3, 5), (6, 4, 6), (5, 4, 6), (7, 5, 7), (6, 5, 7), (8, 6, 8), (7, 6, 8), (9, 7, 9), (8, 7, 9), (0, -1, 2), (3, 0, 3), (1, 0, 3), (4, 1, 4), (2, 1, 4), (5, 2, 5), (3, 2, 5), (6, 3, 6), (4, 3, 6), (7, 4, 7), (5, 4, 7), (8, 5, 8), (6, 5, 8), (9, 6, 9), (7, 6, 9), (0, -1, 3), (4, 0, 4), (1, 0, 4), (5, 1, 5), (2, 1, 5), (6, 2, 6), (3, 2, 6), (7, 3, 7), (4, 3, 7), (8, 4, 8), (5, 4, 8), (9, 5, 9), (6, 5, 9), (0, -1, 4), (5, 0, 5), (1, 0, 5), (6, 1, 6), (2, 1, 6), (7, 2, 7), (3, 2, 7), (8, 3, 8), (4, 3, 8), (9, 4, 9), (5, 4, 9), (0, -1, 5), (6, 0, 6), (1, 0, 6), (7, 1, 7), (2, 1, 7), (8, 2, 8), (3, 2, 8), (9, 3, 9), (4, 3, 9), (0, -1, 6), (7, 0, 7), (1, 0, 7), (8, 1, 8), (2, 1, 8), (9, 2, 9), (3, 2, 9), (0, -1, 7), (8, 0, 8), (1, 0, 8), (9, 1, 9), (2, 1, 9), (0, -1, 8), (9, 0, 9), (1, 0, 9), (0, -1, 9)]
    '''
    for i,Xspan in enumerate(Xspans):
        print('------------------------------')
        print('------------------------------')
        print('STARTING LOOP for Xspan'+str(Xspan))
        print('------------------------------')
        select_k(Xspan, derivations, terminals, forest_d, Xspan_forest_d, parse_probs, rel_probs, rescore_matrix, rescore_config, K)

    ## the goal is to get to the root Xspan(0,-1,length-1) with all hyperedge_ids in it
    ## find 1-best and write out
    best_tree = final_1best(length, derivations, parse_probs, rel_probs)
    to_conllu(out_path, best_tree, sent, tags)

def select_k(Xspan, derivations, terminals, forest_d, Xspan_forest_d, parse_probs, rel_probs, rescore_matrix, rescore_config, K):
    ## eisner_beam_k = {4,8,16,32,64,128}
    X,a,b = Xspan

    '''
    [1] prepare priority queue
    '''
    priq = []
    heapq.heapify(priq) # priq of candidates
    #best_K_buffer = [] # initialize 
    #heapq.heapify(best_K_buffer) # priq-temp

    '''
    [2] prepare incoming edge candidates of node X
    '''
    incoming_edge_ids = Xspan_forest_d[Xspan]
    print('All candidates: '+str(incoming_edge_ids))
    label_for_incoming_edges_d = defaultdict(list)
    for edge_id in incoming_edge_ids:
        sp = edge_id.split('_')
        edge_id_nolabel = '_'.join(sp[:-1])
        label_for_incoming_edges_d[edge_id_nolabel].append(int(sp[-1]))

    '''
    [3] initialize cube based on boundary c
    '''
    visited = set()
    print('[3] initialize cube based on boundary c')
    for edge_id_nolabel in label_for_incoming_edges_d.keys():
        print('------------------------------')
        print('CURRENT_EDGE: '+str(edge_id_nolabel)) #1_1_2_0_1_7
        X,A,B,a,c,b = list(map(int,edge_id_nolabel.split('_')))
        print('X: '+str(X))
        print('A: '+str(A)+' B: '+str(B))
        print('a: '+str(a)+' c: '+str(c)+' b: '+str(b))
        print('------------------------------')

        ## loop based on labels inside cube_next()
        cube_next(derivations, terminals, forest_d, label_for_incoming_edges_d, edge_id_nolabel, Xspan, c, 0, 0, X, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config, X==0)

    '''
    [4] actual cube pruning
    '''
    print('[4] ACTUAL CUBE PRUNING')
    best_K_buffer = []
    cnt=0
    while len(priq) > 0:
        ## cnt is necessary to distinguish hypotheses in heapq, otherwise shows typeerror
        cnt+=1

        '''
        [A] pop next-best from priq
        '''
        neglogp, ka, kb, c, lb, deprel, md, hd, lhs, rhs, edge_id_nolabel, comb_type = heapq.heappop(priq) ## heappop returns 'minimum' (i.e., -maximum)
        logp = -neglogp

        '''
        [B] create new derivation
        '''
        #print(comb_type)
        if comb_type==1:
            edges = {(hd,md)}
            edges_l = {(hd,md,lb,deprel)}
            num_roots = 1 if hd*md==0 else 0
            node_names = set([edge_id_nolabel])
        
        elif comb_type==2:
            edges = {(hd,md)} | rhs.depedges
            edges_l = {(hd,md,lb,deprel)} | rhs.depedges_l
            num_roots = 1+rhs.num_roots if hd*md==0 else 0+rhs.num_roots     
            node_names = rhs.node_names | set([edge_id_nolabel])

        elif comb_type==3:
            edges = lhs.depedges | {(hd,md)}
            edges_l =  lhs.depedges_l | {(hd,md,lb,deprel)}
            num_roots = 1+lhs.num_roots if hd*md==0 else 0+lhs.num_roots
            node_names = lhs.node_names | set([edge_id_nolabel])

        ## combining 2 subderivations
        else: #if comb_type==0:
            edges = lhs.depedges | rhs.depedges
            edges_l = lhs.depedges_l | rhs.depedges_l
            edges.add((hd,md)) # head, tail
            edges_l.add((hd,md,lb,deprel))
            num_roots = lhs.num_roots + rhs.num_roots
            node_names = lhs.node_names | rhs.node_names | set([edge_id_nolabel])
        
        '''
        [C] check validity of new derivation
        '''
        is_violate = (num_roots > 1)
        #print(num_roots)
        has_head = set()
        for edge in edges:
            if edge[1] in has_head:
                is_violate = True
                break
            has_head.add(edge[1])
        
        '''
        [D] insert new derivation to best_K_buffer
        '''
        ## stop if invalid
        ## find position to insert
        j = -1 #init
        for i, best_K in enumerate(best_K_buffer):
            '''legacy
            not applicable to Xspans
            ## hypotheses with same edges should have same logp
            if best_K[-1].depedges == edges:
                is_violate = True
                break
            '''
            if best_K[-1].acclogp < logp:
                j = i
                break

        ## insert new derivation to best_K_buffer
        if not is_violate: 
            new_hyp = HypoD(logp, X, edges_l, edges, node_names, c, num_roots)
            if j == -1:
                best_K_buffer.append([new_hyp.acclogp, cnt, new_hyp])
            else:
                best_K_buffer.insert(j, [new_hyp.acclogp, cnt, new_hyp])

        '''
        [E] check whether to stop loop
        '''
        if comb_type==1:
            break
        if len(best_K_buffer) >= K:
            break

        '''
        [F] move on to next grid
        '''
        cube_next(derivations, terminals, forest_d, label_for_incoming_edges_d, edge_id_nolabel, Xspan, c, ka+1, kb, X, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config, X==0)
        cube_next(derivations, terminals, forest_d, label_for_incoming_edges_d, edge_id_nolabel, Xspan, c, ka, kb+1, X, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config, X==0)

    '''
    [5] keep buffer to D
    '''
    ## fixed: best_K_buffer is heapq. extra sorted() is not necessary.
    ## best_K_buffer = sorted(best_K_buffer, key=lambda x: x.acclogp, reverse=True)
    derivations[Xspan] = best_K_buffer[:K]
    print('********************')
    print(Xspan)
    print(derivations[Xspan])
    print('********************')

def cube_next(derivations, terminals, forest_d, label_for_incoming_edges_d, edge_id_nolabel, Xspan, c, ka, kb, X, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config, is_root):
    '''
    ka: subderivations (Aspans)
    kb: subderivations (Bspans)
    init: ka,ka = 0,0
    '''

    X,A,B,a,c,b = list(map(int,edge_id_nolabel.split('_')))
    print('-----')
    print('k1: '+str(ka)+' k2: '+str(kb))
    print('cube: '+str((ka,kb,A,c,B)))
    if (ka,kb,c) in visited:
        print('visited')
        return
    visited.add((ka,kb,A,c,B))

    for i,lb in enumerate(label_for_incoming_edges_d[edge_id_nolabel]):
        '''
        [1] prepare spans and nodes
        '''
        ## head_A_B_lmost_boundary_rmost_label-id
        incoming_edge_id = edge_id_nolabel+'_'+str(lb)
        he = forest_d[incoming_edge_id]
        deprel,X,A,B, = he['label'],he['head'],he['left_tail'],he['right_tail']
        if (A==B) or (X not in {A,B}):
            return
        tail = A if B==X else B
        X,a,b = Xspan
        Aspan, Bspan = (A,a,c), (B,c,b)
        print('A: '+str(A), 'c: '+str(c), 'B: '+str(B), 'lb: '+str(lb), 'deprel: '+str(deprel))

        '''
        [2] check whether to stop
        '''
        #if (len(derivations[Aspan]) <= ka or len(derivations[Bspan]) <= kb) and (Aspan not in terminals and Bspan not in terminals):
        if (len(derivations[Aspan]) <= ka or len(derivations[Bspan]) <= kb):
            print('cube_end')
            return

        '''
        [3] handle terminals
        '''
        ## merged into [4]

        '''
        [4] compute score for new Xspan
        '''
        print('Aspan: '+str(Aspan)+' + Bspan: '+str(Bspan))
        md,hd = tail,X
        if Aspan in terminals and Bspan in terminals:
            comb_type=1
            lhs, rhs = None, None
            if rescore_config['RESCORE']=='True':
                newlogp = bert_rescore(comb_type, rhs, lhs, md, hd, lb, rescore_matrix, rescore_config)
            else: #for debug
                newlogp = np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)            
            print('new: '+str(np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)))

        elif Aspan in terminals:
            comb_type=2
            lhs, rhs = None, derivations[Bspan][kb][-1]
            if rescore_config['RESCORE']=='True':
                newlogp = bert_rescore(comb_type, rhs, lhs, md, hd, lb, rescore_matrix, rescore_config)
            else: #for debug
                newlogp = rhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
            print('new: '+str(np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10))+' +Bspan: '+str(rhs.acclogp))

        elif Bspan in terminals:
            comb_type=3
            lhs, rhs = derivations[Aspan][ka][-1], None
            if rescore_config['RESCORE']=='True':
                newlogp = bert_rescore(comb_type, rhs, lhs, md, hd, lb, rescore_matrix, rescore_config)
            else: #for debug
                newlogp = lhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
            print('Aspan: '+str(lhs.acclogp)+' + new: '+str(np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)))

        else:
            comb_type=0
            lhs, rhs= derivations[Aspan][ka][-1], derivations[Bspan][kb][-1]
            if rescore_config['RESCORE']=='True':
                newlogp = bert_rescore(comb_type, rhs, lhs, md, hd, lb, rescore_matrix, rescore_config)
            else: #for debug
                newlogp = lhs.acclogp + rhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
            print('Aspan: '+str(lhs.acclogp)+' + Bspan: '+str(rhs.acclogp))
    
        '''
        [5] add bert score
        '''
        print('newlogp: '+str(newlogp))
        print('----------')

        '''
        [6] push to priq
        '''
        heapq.heappush(priq, [-newlogp, ka, kb, c, lb, deprel, md, hd, lhs, rhs, edge_id_nolabel, comb_type])

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (C) Rescoring Function
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def bert_rescore(comb_type, rhs, lhs, md, hd, lb, rescore_matrix, rescore_config):
    '''BERT score integration
    Apply rescoring function when condition is met
    condition: md and hd are both verbs

    alpha: weight
    beta: bias
    '''
    alpha, beta = rescore_config['alpha'], rescore_config['beta']
    #print('Xspan: ' + str(Xspan))
    ## rescoring condition
    if comb_type==1:
        if rescore_matrix[md-1] is not None:
            return np.log(rescore_matrix[md-1][hd-1]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
        return np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)    
    
    elif comb_type==2:
        if rescore_matrix[md-1] is not None:
            return rhs.acclogp + np.log(rescore_matrix[md-1][hd-1]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
        return rhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
    elif comb_type==3:
        if rescore_matrix[md-1] is not None:
            return lhs.acclogp + np.log(rescore_matrix[md-1][hd-1]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
        return lhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
    else:
        if rescore_matrix[md-1] is not None:
            return lhs.acclogp + rhs.acclogp + np.log(rescore_matrix[md-1][hd-1]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)
        return lhs.acclogp + rhs.acclogp + np.log(parse_probs[md,hd]+1e-10) + np.log(rel_probs[md,hd,:][lb]+1e-10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', help='dir path for parsed data (.plk)')
    parser.add_argument('--out_path', help='1best tree output file path (.conllu)')
    parser.add_argument('--rescore_cfg', help='file path for rescore config (.cfg)')
    parser.add_argument('--rescore', default='False', type=str, help='rescore on cube pruning or not')
    parser.add_argument('--test', default='False', type=str, help='test or not')
    parser.add_argument('--K', type=int, help='K for Kbest')
    parser.add_argument('--alpha', type=float, help='alpha weight for scoring function')
    parser.add_argument('--beta', type=float, help='beta bias for scoring function')
    args = parser.parse_args()

    pkl_dir = args.pkl_dir
    out_path = args.out_path
    rescore_cfg = args.rescore_cfg
    test = args.test
    rescore_config = {'alpha': args.alpha,
                    'beta': args.beta,
                    'RESCORE': args.rescore}
    print('TEST MODE: '+str(test))

    print('0: Checking Existing Files')
    if os.path.exists(out_path):
        sys.exit('File: '+out_path+' Exists. Already Searched with these Parameters.')

    print('1: Loading Parsed Forests')
    all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_loader(pkl_dir)
    
    cnt=0

    ## for 1-best search on full .conllu file
    if test=='False':
        print('2: Loading Configurations')
        if not os.path.exists(rescore_cfg):
            sys.exit('rescore.cfg Not Found')
        cfg = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
        cfg.read(rescore_cfg)
        
        print('3: Loading Rescoring Model')
        remodel = RescoreModel(cfg)

        print('4: Starting 1-best Search on Forests')
        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            rescore_matrix = remodel.head_prediction(sent, tags)
            main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, sent, tags, out_path, args.K)
            cnt+=1
        
        print('5: Done')

    ## for testing purpose using an example from devset
    else:
        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            if cnt==1:
                break

            rescore_matrix = [None, None, None, None, np.array([4.8442665e-08, 4.3052935e-08, 4.3104702e-08, 4.2075676e-08,
        4.6197080e-08, 5.0632288e-08, 1.3114438e-07, 4.5686441e-08,
        9.9988854e-01, 9.7648467e-08, 5.0003713e-08, 1.1035899e-04,
        4.6682160e-08, 4.3503015e-08, 3.9373546e-08, 4.4436916e-08,
        5.2203276e-08, 3.8286085e-08, 4.1419408e-08, 4.2155246e-08,
        5.9530176e-08, 3.0217173e-08], dtype=np.float32), None, None, None, np.array([8.3981213e-05, 5.8226364e-05, 8.3306048e-05, 8.6549109e-05,
        6.3370311e-01, 5.6491252e-05, 1.2038154e-04, 7.4531978e-05,
        1.4989292e-02, 1.6915977e-04, 9.4180687e-05, 3.4962612e-01,
        8.5302905e-05, 8.8232744e-05, 7.6945347e-05, 8.5135165e-05,
        9.3006085e-05, 7.2522649e-05, 7.9221318e-05, 9.1847163e-05,
        1.2829319e-04, 5.4189884e-05], dtype=np.float32), None, None, np.array([2.7965630e-06, 2.1852975e-06, 3.0401338e-06, 2.9491682e-06,
        4.5953339e-04, 1.9971540e-06, 3.6995973e-06, 3.2637997e-06,
        9.9940789e-01, 3.4897778e-06, 3.4665786e-06, 7.5384894e-05,
        3.0627450e-06, 3.2644596e-06, 2.7905899e-06, 3.1082448e-06,
        3.2536063e-06, 2.6948662e-06, 2.9266587e-06, 3.2492653e-06,
        3.7590371e-06, 2.1705341e-06], dtype=np.float32), None, None, None, None, None, None, None, None, np.array([5.8608354e-08, 4.7363333e-08, 6.1556136e-08, 6.0390697e-08,
        4.2202023e-06, 4.4770150e-08, 8.1334115e-08, 5.9691686e-08,
        9.3881460e-04, 9.5667900e-08, 8.2211741e-08, 9.9905533e-01,
        2.0578375e-07, 1.3435931e-07, 8.6339938e-08, 7.1877118e-08,
        5.5844964e-08, 5.8704348e-08, 6.7323739e-08, 7.6822495e-08,
        7.8168952e-08, 4.3558558e-08], dtype=np.float32), None]
            print(len(forest['hyperedges']))

            #verb_nodes = [bool(tag[0] == 'V') for tag in tags]

            ## json
            forest_out = 'test/test_forest.json'
            with open(forest_out, 'w') as f:
                json.dump(forest, f)

            main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, sent, tags, out_path, args.K)
            cnt+=1

            print(sent)
            print(tags)
            #print(parse_probs)
            #print(rel_probs.shape)

