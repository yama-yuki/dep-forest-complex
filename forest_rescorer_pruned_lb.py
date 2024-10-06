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

from logging import getLogger, config
with open('log.cfg', 'r') as f:
    log_cfg = json.load(f)
config.dictConfig(log_cfg)
logger = getLogger(__name__)

from rescore_module.my_lib.rescore_all import RescoreModel
from lib.conll import final_1best_label, to_conllu_label, pkl_loader, rels_loader, pruned_loader


class HypoD:
    '''
    hypothesis for each Xspan, which will be sorted in derivation D
    '''
    def __init__(self, acclogp, X_id, depedges, num_roots):
        self.acclogp = acclogp
        self.X_id = int(X_id)
        self.depedges = depedges
        self.num_roots = num_roots

    def to_list(self):
        print(str(self.acclogp))
        print(str(self.X_id))
        print(list(self.depedges))

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

def load_pruned(forest):
    node2edges = defaultdict(list)
    id2Xspan = defaultdict()
    Xspan2id = defaultdict()
    topological_id = []

    cur_id = None
    
    for i,line in enumerate(forest):
        l = len(line)
        ##['0', "ROOT No , it was n't Black Monday ."] 2
        ##['17'] 1
        ##['0', ' 0 [0-1]', '0 ||| ', ''] 4
        if l==2:
            forest_id, sent = int(line[0]), line[1]
            continue
        elif l==1:
            node_len = int(line[0])
            continue

        ## actual nodes and hyperedges
        ## nodes
        ## '19	 1-3 [1-3]	1 ||| 	'
        if l==4:
            X_id, X_name, n_hyperedges = int(line[0]), line[1][1:], int(line[2].split(' ')[0])
            head_n_label, span_range = X_name.split(' ') ##'[0-1]'
            X = list(map(int,head_n_label.split('-')))
            a,b = list(map(int,span_range[1:-1].split('-')))
            if n_hyperedges==0 and len(X)==1:
                ## terminal
                Xspan = (X[0], a, b)
            else:
                ## non-terminal
                X,lb = X
                ##Xspan = (X, a, b, lb)
                Xspan = (X, a, b)
            cur_id = X_id
            id2Xspan[X_id]=Xspan
            Xspan2id[Xspan]=X_id
            topological_id.append(X_id)

        ## hyperedges
        ## '	1 2 ||| 	0=0.0033416455000000  '
        elif l==3:
            logp = float(line[2].split('0=')[1].split(' ')[0])
            A_id, B_id = list(map(int,line[1].split(' ')[:2]))
            node2edges[cur_id].append((A_id,B_id,logp))
        
        else:
            continue

    return node2edges, id2Xspan, Xspan2id, topological_id

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

    logger.debug('Topologically Sorted Xspans: '+ str(Xspans))

    return Xspans

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (B) Cube Pruning Algorithm
## This algorithm searches Kbest derivations of a Xspan which is a triplet of X(head node), a(leftmost governing span boundary of X), and b(rightmost governing span boundary of X)
## Then returns derivation[(0,0,length][0], the resulting 1best dependency tree with a root node governing leftmost to rightmost
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def cube(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, out_path, K, cnt, debug=False):

    length = len(sent)
    logger.debug('sentence length: '+str(length))

    ## get sorted list of nodes from a parsed forest
    node2edges, id2Xspan, Xspan2id, topological_id = load_pruned(forest)
    logger.debug('total nodes: '+str(len(topological_id)))

    cnt=0
    for node in node2edges:
        cnt+=len(node2edges[node])
    n_node = len(topological_id)-length
    n_edge = cnt
    ##return n_node, n_edge, n_edge/n_node, length
    ##sys.exit()

    ## initialize
    terminals = set()
    derivations = defaultdict(list)

    ## list of Xspans
    Xspans = []
    for X_id in topological_id:
        Xspan = id2Xspan[X_id]
        if Xspan[2]-Xspan[1]==1:
            terminals.add(Xspan)
        Xspans.append(Xspan)
    logger.debug('Xspans: '+str(Xspans))

    ## initialize derivations
    for terminal_Xspan in terminals:
        logp = 0.0
        X,a,b = terminal_Xspan
        X_id = Xspan2id[terminal_Xspan]
        new_hyp = HypoD(logp, X_id, set(), 1 if X==0 else 0)
        derivations[terminal_Xspan].append(new_hyp)
        ## new_hyp = HypoD(acclogp, X_id, A_id, B_id, depedges, num_roots)
    logger.debug('Inialized terminal D: '+str(derivations))

    ## main loop
    '''Bottom-up combinatory operation of span flagments
    e.g., Aspan(0,0,1) + Bspan(1,1,9) -> Xspan(0,0,9)
    for vertex in topological order
    '''
    for X_id in topological_id:
        Xspan = id2Xspan[X_id]
        logger.debug('------------------------------')
        logger.debug('------------------------------')
        logger.debug('STARTING LOOP for Xspan'+str(Xspan))
        logger.debug('------------------------------')
        ## skip terminal nodes
        if Xspan in terminals:
            logger.debug('terminal span: '+str(Xspan))
            continue
        kbest(X_id, Xspan, derivations, terminals, node2edges, id2Xspan, Xspan2id, topological_id, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, K, cnt)

    ## the goal is to get to the root Xspan(0,0,length) with all heads in it
    ## find 1-best and write out
    if debug==True:
        return
    best_tree = final_1best_label(length, derivations, parse_probs, rel_probs)

    '''
    pickled relation vocab
    hard coded. to be fixed.
    '''
    deprel_path = '/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl/deprel.pkl'
    to_conllu_label(out_path, best_tree, sent, tags, deprel_path)

def kbest(X_id, Xspan, derivations, terminals, node2edges, id2Xspan, Xspan2id, topological_id, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, K, cnt):

    '''
    [1] prepare priority queue 'heap'
    '''
    logger.debug('[1] INITIALIZE HEAP')
    ## prioritize smaller value
    heap = []
    heapq.heapify(heap)

    '''
    [2] for each incoming edge, add 1best subderivation combination to 'heap'
    '''
    logger.debug('[2] ADD 1BEST CANDIDATES')
    incoming_edges = node2edges[X_id]
    logger.debug('All candidates: '+str(incoming_edges))

    visited = set()
    for incoming_edge in incoming_edges:
        logger.debug('------------------------------')
        logger.debug('CURRENT_EDGE: '+str(incoming_edge))         
        A_id, B_id, logp = incoming_edge ##ids
        Aspan, Bspan = id2Xspan[A_id], id2Xspan[B_id] ##spans
        D_a, D_b = derivations[Aspan], derivations[Bspan] ##subderivations

        X,a,b = Xspan
        A,a,c = Aspan
        B,c,b = Bspan
        unit = X,A,B,a,c,b
        logger.debug('X: '+str(X))
        logger.debug('A: '+str(A)+' B: '+str(B))
        logger.debug('a: '+str(a)+' c: '+str(c)+' b: '+str(b))
        logger.debug('------------------------------')

        ka,kb = 0,0
        ## add 1best subderivation combination
        pushsucc(heap, derivations, terminals, parse_probs, 0, 0, unit, visited)
        ##heapq.heappush(heap,(logp,ka,kb,D_a,D_b,unit)) ##acclogp&hypos
    
    '''
    [3] cube pruning, move to next grid
    '''
    logger.debug('[3] ACTUAL CUBE PRUNING')
    best_K_buffer = []
    while len(heap)>0:
        ## extract next best
        logger.debug(heap)
        neglogp, items = heapq.heappop(heap)
        logp = -neglogp
        ka, kb, md, hd, unit, comb_type, lhs, rhs, lb = items

        '''
        [B] create new derivation
        '''
        if comb_type==1:
            edges = {(hd,md,lb)}
            num_roots = 1 if hd*md==0 else 0
        
        elif comb_type==2:
            edges = {(hd,md,lb)} | rhs.depedges
            num_roots = 1+rhs.num_roots if hd*md==0 else 0+rhs.num_roots     

        elif comb_type==3:
            edges = lhs.depedges | {(hd,md,lb)}
            num_roots = 1+lhs.num_roots if hd*md==0 else 0+lhs.num_roots

        ## combining 2 subderivations
        else: #if comb_type==0:
            edges = lhs.depedges | rhs.depedges
            edges.add((hd,md,lb)) # head, tail
            num_roots = lhs.num_roots + rhs.num_roots

        ## make new hypo
        newhypo = HypoD(logp, X_id, edges, num_roots)

        ## add new hypo to buffer
        best_K_buffer.append(newhypo)

        ## next grid
        pushsucc(heap, derivations, terminals, parse_probs, ka+1, kb, unit, visited)
        pushsucc(heap, derivations, terminals, parse_probs, ka, kb+1, unit, visited)

        ## stop when buffer is filled
        if len(best_K_buffer)>=K:
            break

    '''
    [4] sort buffer to derivations
    '''
    ## sort buffer to D
    best_K_buffer = sorted(best_K_buffer, key=lambda x: x.acclogp, reverse=True)
    derivations[Xspan] = best_K_buffer[:K]
    logger.debug('********************')
    logger.debug(Xspan)
    logger.debug(derivations[Xspan])
    logger.debug('********************')

def pushsucc(heap, derivations, terminals, parse_probs, ka, kb, unit, visited):
    '''
    ka: subderivations (Aspans)
    kb: subderivations (Bspans)
    init: ka,ka = 0,0
    '''
    X,A,B,a,c,b = unit
    logger.debug('-----')
    logger.debug('k1: '+str(ka)+' k2: '+str(kb))
    logger.debug('AcB: '+str((A,c,B)))
    if (ka,kb,unit) in visited:
        logger.debug('visited')
        return
    visited.add((ka,kb,unit))

    '''
    [1] prepare spans and nodes
    '''
    if (A==B) or (X not in {A,B}):
        return
    tail = A if B==X else B
    Aspan, Bspan = (A,a,c), (B,c,b)

    '''
    [2] check whether to stop
    '''
    if (len(derivations[Aspan]) <= ka or len(derivations[Bspan]) <= kb):
        logger.debug('cube_end')
        return

    '''
    [3] compute score for new Xspan
    '''
    logger.debug('Aspan: '+str(Aspan)+' + Bspan: '+str(Bspan))
    md,hd = tail,X
    if Aspan in terminals and Bspan in terminals:
        lhs, rhs = None, None
        logp = np.log(parse_probs[md,hd]+1e-10)
        comb_type=1
        logger.debug('new: '+str(np.log(parse_probs[md,hd]+1e-10)))
    elif Aspan in terminals:
        lhs, rhs = None, derivations[Bspan][kb]
        logp = rhs.acclogp + np.log(parse_probs[md,hd]+1e-10)
        comb_type=2
        logger.debug('new: '+str(np.log(parse_probs[md,hd]+1e-10))+' +Bspan: '+str(rhs.acclogp))
    elif Bspan in terminals:
        lhs, rhs = derivations[Aspan][ka], None
        logp = lhs.acclogp + np.log(parse_probs[md,hd]+1e-10)
        comb_type=3
        logger.debug('Aspan: '+str(lhs.acclogp)+' + new: '+str(np.log(parse_probs[md,hd]+1e-10)))
    else:
        lhs, rhs= derivations[Aspan][ka], derivations[Bspan][kb]
        logp = lhs.acclogp + rhs.acclogp + np.log(parse_probs[md,hd]+1e-10)
        comb_type=0
        logger.debug('Aspan: '+str(lhs.acclogp)+' + Bspan: '+str(rhs.acclogp))

    ##np.log(rel_probs[md,hd,:][lb]+1e-10)
    '''
    for i, logp in enumerate(new_las):
        if i not in (Vocab.PAD, Vocab.ROOT, Vocab.UNK):
            las_logp = uas_logp + logp
            heapq.heappush(priq, (-las_logp,u,k1,k2,i))
    '''
    ## finding best label
    las = rel_probs[md,hd,:]
    las_l = [logp for logp in las]
    max_las = max(las_l)
    lb = las_l.index(max_las)

    '''
    [5] add BERT score
    '''
    logger.debug('logp: '+str(logp))
    if rescore_config['RESCORE']=='True':
        newlogp = rescore(logp,md,hd,rescore_matrix_V,rescore_matrix_NV,rescore_config)
    else: #for debug
        newlogp = logp
    
    logger.debug('newlogp: '+str(newlogp))
    logger.debug('----------')

    '''
    [6] push to heap
    '''
    items = (ka, kb, md, hd, unit, comb_type, lhs, rhs, lb)
    if not items or not newlogp:
        return
    heapq.heappush(heap, [-newlogp, items])

def rescore(logp,md,hd,rescore_matrix_V,rescore_matrix_NV,rescore_config):
    '''BERT score integration
    Apply rescoring function when condition is met
    condition: md and hd are both verbs
    alpha: weight
    beta: bias
    '''
    alpha, beta = rescore_config['alpha'], rescore_config['beta']
    rescore_mode = rescore_config['rescore_mode']
    ## VERB rescoring conditions
    if rescore_mode=='A':
        if rescore_matrix_V[md-1] is not None: # parent node is verb
            bert_score = np.log(rescore_matrix_V[md-1][hd-1])
            return logp + beta + alpha*bert_score
        else:
            bert_score = np.log(rescore_matrix_NV[md-1][hd-1])
            return logp + beta + alpha*bert_score
    elif rescore_mode=='V':
        if rescore_matrix_V[md-1] is not None: # parent node is verb
            bert_score = np.log(rescore_matrix_V[md-1][hd-1])
            return logp + beta + alpha*bert_score
        else:
            return logp
    elif rescore_mode=='N':
        if rescore_matrix_V[md-1] is not None: # parent node is verb
            return logp
        else:
            bert_score = np.log(rescore_matrix_NV[md-1][hd-1])
            return logp + beta + alpha*bert_score
    else:
        sys.exit('specify rescore_mode')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, help='dir path for parsed data (.pkl)')
    parser.add_argument('--pruned_dir', type=str, help='dir for pruned forest (.forest)')
    parser.add_argument('--out_path', type=str, help='1best tree output file path (.conllu)')

    parser.add_argument('--rescore_v_cfg', type=str, help='file path for rescore config (.cfg)')
    parser.add_argument('--rescore_nv_cfg', type=str, help='file path for rescore config (.cfg)')
    parser.add_argument('--rescore', type=str, default='False', help='rescore on cube pruning or not')

    parser.add_argument('--K', type=int, default=3, help='K for Kbest')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha weight for scoring function')
    parser.add_argument('--beta', type=float, default=0.1, help='beta bias for scoring function')

    parser.add_argument('--from_saved', default='False', type=str)
    parser.add_argument('--save_rescore', default='False', type=str)
    
    parser.add_argument('--v_path', type=str, help='vmodel pkl')
    parser.add_argument('--nv_path', type=str, help='nvmodel pkl')

    parser.add_argument('--rel_vocab_path', type=str, help='path to rels.txt')
    parser.add_argument('--rescore_mode', type=str, help='V:verbial,N:non-verbial,A:all')
    args = parser.parse_args()
    #sys.exit('hoge1')

    pkl_dir = args.pkl_dir
    pruned_dir = args.pruned_dir
    rescore_v_cfg = args.rescore_v_cfg
    rescore_nv_cfg = args.rescore_nv_cfg
    rescore_config = {'alpha': args.alpha,
                    'beta': args.beta,
                    'RESCORE': args.rescore,
                    'rescore_mode':args.rescore_mode}
    
    ##-------------------------------------------------------------------------##
    ## LOAD DATA

    logger.debug('0: Checking Existing Files')
    if os.path.exists(args.out_path):
        sys.exit('File: '+args.out_path+' Exists. Already Searched with these Parameters.')

    logger.debug('1: Loading Parsed Forests')
    _,all_parse_probs,all_rel_probs,all_sents,all_tags = pkl_loader(pkl_dir)
    all_forests = pruned_loader(pruned_dir)

    rel_vocabs = rels_loader(args.rel_vocab_path)
    logger.debug('Done Loading Files')

    ##-------------------------------------------------------------------------##
    ## SAVING MODEL PREDICTION

    ## Predict using models and save as pkl
    if args.save_rescore=='True':
        logger.info('2: save_rescore')
        logger.info('3: Loading Model Configurations')
        if not os.path.exists(rescore_v_cfg):
            sys.exit('rescore_v.cfg Not Found')
        if not os.path.exists(rescore_nv_cfg):
            sys.exit('rescore_nv.cfg Not Found')        
        cfg_v = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
        cfg_v.read(rescore_v_cfg)
        cfg_nv = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
        cfg_nv.read(rescore_nv_cfg)

        remodel_V = RescoreModel(cfg_v)
        remodel_NV = RescoreModel(cfg_nv)

        logger.info('4: Predicting V')
        rescore_matrix_V = [remodel_V.head_prediction_V(sent, tags) for sent,tags in tqdm(zip(all_sents,all_tags))]
        with open(args.v_path, 'wb') as pv:
            pkl.dump(rescore_matrix_V, pv)        
        rescore_matrix_V = []

        logger.info('5: Predicting NV')
        rescore_matrix_NV = [remodel_NV.head_prediction_NV(sent, tags) for sent,tags in tqdm(zip(all_sents,all_tags))]
        with open(args.nv_path, 'wb') as pnv:
            pkl.dump(rescore_matrix_NV, pnv)     
        rescore_matrix_NV = []

        logger.info('6: Head Prediction Done & Saved')
        sys.exit()

    ##-------------------------------------------------------------------------##
    ## CUBE PRUNING with BERT

    cnt=0
    ## for 1-best search on full .conllu file
    ## for testing purpose using an example from devset
    n,e,p,t = 0,0,0,0

    if args.rescore=='True':
        logger.debug('2: rescore')

        if args.from_saved=='True':
            logger.debug('3: Loading Saved Predictions')
            with open(args.v_path, 'rb') as pv:
                rescore_matrix_V_saved = pkl.load(pv)
            with open(args.nv_path, 'rb') as pnv:
                rescore_matrix_NV_saved = pkl.load(pnv)

            logger.debug('4: Do Cube Pruning')
            for i,(forest,parse_probs,rel_probs,sent,tags) in enumerate(tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags))):
                rescore_matrix_V = rescore_matrix_V_saved[i]
                rescore_matrix_NV = rescore_matrix_NV_saved[i]
                ##node, edge, ratio, tokens = 
                cube(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, args.K, cnt)
                ##(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, K)
                cnt+=1
            logger.debug('5: Done')
        
        else:
            logger.debug('3: Loading Model Configurations')
            if not os.path.exists(rescore_v_cfg):
                sys.exit('rescore_v.cfg Not Found')
            if not os.path.exists(rescore_nv_cfg):
                sys.exit('rescore_nv.cfg Not Found')        
            cfg_v = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
            cfg_v.read(rescore_v_cfg)
            cfg_nv = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
            cfg_nv.read(rescore_nv_cfg)
            
            logger.debug('4: Loading Rescoring Models')
            remodel_V = RescoreModel(cfg_v)
            remodel_NV = RescoreModel(cfg_nv)

            logger.debug('5: Do Cube Pruning')
            for i,(forest,parse_probs,rel_probs,sent,tags) in enumerate(tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags))):
                rescore_matrix_V = remodel_V.head_prediction_V(sent, tags)
                rescore_matrix_NV = remodel_NV.head_prediction_NV(sent, tags)
                ##node, edge, ratio, tokens = 
                cube(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, args.K, cnt)
                ##(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, K)
                cnt+=1
            logger.debug('6: Done')
    
    ##-------------------------------------------------------------------------##
    ## CUBE PRUNING without BERT

    else:
        logger.debug('2: vanilla')
        logger.debug('3: Do Cube Pruning')
        for i,(forest,parse_probs,rel_probs,sent,tags) in enumerate(tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags))):
            rescore_matrix_V = None
            rescore_matrix_NV = None
            ##node, edge, ratio, tokens = 
            cube(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, args.K, cnt, debug=True)
            ##(forest, parse_probs, rel_probs, rescore_matrix_V, rescore_matrix_NV, rescore_config, sent, tags, args.out_path, K)
            cnt+=1
            logger.debug('4: Done')
            '''
            n+=node
            e+=edge
            p+=ratio
            t+=tokens
            '''
    ##-------------------------------------------------------------------------##

