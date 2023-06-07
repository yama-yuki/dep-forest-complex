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
from lib.conll import to_conllu, Tree

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (A) Forest Reader
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

## derivation D
class HypoD:
    ## tree hypotheses
    def __init__(self, node, acclogp, hyperedges, depedges, b, num_roots):
        self.node = int(node)
        self.acclogp = acclogp
        self.hyperedges = hyperedges
        self.depedges = depedges
        self.b = b
        self.num_roots = num_roots

    def to_list(self):
        print(str(self.node))
        print(list(self.hyperedges))
        print(list(self.depedges))
        print(str(self.acclogp))

def node2hyperedge(forest):
    '''
    forest_d[node_name]: hyperedge
    '''

    forest_d = defaultdict()
    for he in forest['hyperedges']:
        forest_d[he['name']] = he

    return forest_d

def nodehlr2hyperedge(forest_d, nodes):
    '''
    nodes: head, lmost, rmost

    hlr_forest_d: dict
                  dict[node] = hyperedge_node_name
    '''
    hlr_forest_d = defaultdict(list)
    for node in nodes:
        head, lmost, rmost = node
        for he_name in forest_d:
            sp = list(map(int,he_name.split('_')))
            if sp[0] == head and sp[3] == lmost and sp[5] == rmost:
                hlr_forest_d[node].append(he_name)
            
    return hlr_forest_d

def load_forest(forest):
    ## a list of node_ids
    nodes_hlrlbr = forest['nodes']
    ## node_id->hyperedge
    forest_d = node2hyperedge(forest)
    #print(forest_d)    
    ## bottom-up search on hyperedges based on span_len
    nodes_hlr = topological_sort(nodes_hlrlbr)
    ## node_span_range->[hyperedges]
    hlr_forest_d = nodehlr2hyperedge(forest_d, nodes_hlr)

    return nodes_hlrlbr, nodes_hlr, forest_d, hlr_forest_d

def topological_sort(nodes):
    ## sort nodes based on span length (rising order)
    #9_3_9_0_4_9
    #head_lt_rt_lmost_bound_rmost
    d = defaultdict()
    for node in nodes:
        sp = list(map(int,node.split('_')))
        span_len = sp[5]-sp[3] 
        d[node] = span_len
    sorted_nodes = [node for (node,_) in sorted(d.items(), key=lambda x: x[1])]

    nodes_hlr = []
    for node in sorted_nodes:
        h,_,_,l,_,r,_ = list(map(int,node.split('_')))
        if (h,l,r) not in nodes_hlr:
            nodes_hlr.append((h,l,r)) ## head,lmost,rmost

    print('topological sort')
    print(nodes_hlr)

    return nodes_hlr

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (B) Cube Pruning Algorithm
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, K):
    length = len(forest['node_ids'])
    print('words: '+str(length-1))
    print('with root: '+str(length))
    print('total nodes: '+str(len(forest['nodes'])))

    nodes_hlrlbr, nodes_hlr, forest_d, hlr_forest_d = load_forest(forest)
    #parse_score_d = parse_score(forest_d, length)
    '''
    nodes_hlrlbr: head, ltail, rtail, lmost, boundary, rmost
    nodes_hlr: head, lmost, rmost
    forest_d: node_id -> hyperedge
    hlr_forest_d: span_range(head, lmost, rmost) -> [node_ids]
    '''

    ## create a chart of derivations
    derivations = defaultdict(list)
    ## (1,0,1), ... (9,0,9), (1,1,2)
    
    ## initialize with unary hyperedges
    for x in range(length): ## head loop
        for left in range(length): ## left tail loop
            for right in range(length): ## right tail loop
                #if alpha < x <= beta: 
                if right-left==1 and x==right:
                    derivations[(x,left,right)].append(HypoD(x, 'unary', set(), set(), -1, 0))

    print(derivations)

    ## main loop
    for i,node in enumerate(nodes_hlr):
        #nodes: [(2, 0, 2), (3, 2, 4), (5, 4, 6)] (head, lmost, rmost)
        print('start_loop')
        print(node)
        select_k(i, node, derivations, forest_d, hlr_forest_d, parse_probs, rel_probs, rescore_matrix, rescore_config, K)

    ## the goal is to get to the root HE(0,0,length-1) with all node_ids in it
    print('final_derivation')
    #final = derivations[(0,-1,length-1)]
    #print(final)

    to_conllu(length, derivations, parse_probs, rel_probs)

def select_k(i, node, derivations, forest_d, hlr_forest_d, parse_probs, rel_probs, rescore_matrix, rescore_config, K):
    ## eisner_beam_k = {8,16,32,64,128}
    ## create priq from each incoming hyperedge
    head, lmost, rmost = node

    priq = []
    heapq.heapify(priq) # priq of candidates
        
    ## initialize priority queue
    best_K_buffer = []
    heapq.heapify(best_K_buffer) # priq-temp
    visited = set()

    ## incoming edges
    incoming_edge_ids = hlr_forest_d[node]
    incoming_edge_sp = [tuple(map(int, edge_id.split('_'))) for edge_id in incoming_edge_ids]
    #[(4, 4, 6, 3, 5, 6), (4, 4, 5, 3, 4, 6)]
    print(incoming_edge_ids)
    ## for each incoming edge
    for incoming_edge in incoming_edge_ids:
        print('CURRENT_EDGE: '+str(incoming_edge))
        sp = tuple(map(int, incoming_edge.split('_')))
        he = forest_d[incoming_edge]
        head,left_tail,right_tail, = he['head'],he['left_tail'],he['right_tail']
        tail = left_tail if right_tail==head else right_tail
        #probs = he['prob']
        b, lb = sp[4], sp[6]

        ## create a cube
        lspan, rspan = (left_tail,lmost,b), (right_tail,b,rmost) ##todo: add hypo for subderivations
        #print('b: '+str(b))
        #print('lspan: '+str(lspan))
        #print('rspan: '+str(rspan))

        ## init cube with kl=kr=0, b, lspan, rspan
        cube_next(derivations, node, 0, 0, b, lb, head, lspan, rspan, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config)

    ## actual cube pruning
    while len(priq) > 0: 
        ## extract next-best
        neglogp, kl, kr, b, lb, head, lspan, rspan = heapq.heappop(priq) ## return minimum = -maximum
        logp = -neglogp

        #if not is_root:
        lhs, rhs = derivations[lspan][kl], derivations[rspan][kr]
        edges = lhs.depedges | rhs.depedges
        edges_l = lhs.hyperedges | rhs.hyperedges
        tail = lhs.node if rhs.node==head else rhs.node
        edges.add((head, tail))#head->tail
        edges_l.add((head, tail, lb))
        num_roots = lhs.num_roots + rhs.num_roots

        '''
        else:
            rhs = derivations[rspan][kr]
            edges = rhs.depedges
            he = rhs.hyperedges
            tail = rhs.node
            edges.add((head, tail))
            num_roots = rhs.num_roots
        '''

        ### check if violates
        is_violate = (num_roots > 1)
        has_head = set()
        for edge in edges:
            if edge[1] in has_head:
                is_violate = True
                break
            has_head.add(edge[1])
        j = -1
        
        ## append item to buffer
        #heapq.heappush(best_K_buffer, [logp, node])

        for i, hyp in enumerate(best_K_buffer):
            ## hypotheses with same edges should have same logp
            if hyp.depedges == edges:
                is_violate = True
                break
            if hyp.acclogp < logp:
                j = i
                break

        ### insert
        if not is_violate :
            acclogp = logp
            new_hyp = HypoD(head, acclogp, edges_l, edges, b, num_roots)
            if j == -1:
                best_K_buffer.append(new_hyp)
            else:
                best_K_buffer.insert(j, new_hyp)

        if len(best_K_buffer) >= K:
            print('best_K')
            print(best_K_buffer)
            break

        ## move on to next grid
        cube_next(derivations, node, kl+1, kr, b, lb, head, lspan, rspan, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config)
        cube_next(derivations, node, kl, kr+1, b, lb, head, lspan, rspan, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config)

    ## sort buffer to D(v)
    ## best_K_buffer is heapq. extra sorted() is not necessary.
    #best_K_buffer = sorted(best_K_buffer, key=lambda x: x.acclogp, reverse=True)
    derivations[node] = best_K_buffer[:K]

def cube_next(derivations, node, kl, kr, b, lb, head, lspan, rspan, visited, priq, parse_probs, rel_probs, rescore_matrix, rescore_config):
    '''
    (lhs_list, rhs_list, visited, priq,
        is_making_incomplete, u, k1, k2, new_uas, new_las, is_s_0 = False)
    
    lhs_list: candidates of left_tails; alpha
    rhs_list: candidates of right_tails; beta
    b: boundary; gamma

    k1: x-axis; lhs
    k2: y-axis; rhs
    init:
        k1,k2 = 0,0
    '''

    print('k1: '+str(kl)+' k2: '+str(kr))
    ## look for next block
    #is_root = bool(lspan[:2] == (0,-1))
    if (len(derivations[lspan]) <= kl or len(derivations[rspan]) <= kr):# and not is_root:
        #print(derivations[lspan])
        #print(derivations[rspan])
        print('cube_end')
        return

    '''
    if (kl,kr,b) in visited:
        print(((kl,kr,b) in visited))
        print('visited')
        return
    '''
    visited.add((kl,kr,b))
    print('cube: '+str((kl,kr,b)))

    #if not is_root:
    ## derivations[lspan]/[rspan]
    lhs, rhs = derivations[lspan][kl], derivations[rspan][kr]
    #print(derivations)
    #print(lspan)
    #print(rspan)
    l_tail, r_tail = lhs.node, rhs.node
    if l_tail == r_tail or head not in {l_tail, r_tail}:
        #print(l_tail, r_tail)
        return
    tail = l_tail if r_tail==head else r_tail
    print('l_tail: '+str(l_tail), 'r_tail: '+str(r_tail))
    print(lhs.acclogp)
    print(rhs.acclogp)

    mi,hi = tail,head
    if lhs.acclogp=='unary' and rhs.acclogp=='unary':
        logp = np.log(parse_probs[mi,hi]+1e-10) + np.log(rel_probs[mi,hi,:][lb]+1e-10)
    elif lhs.acclogp=='unary':
        logp = rhs.acclogp + np.log(parse_probs[mi,hi]+1e-10) + np.log(rel_probs[mi,hi,:][lb]+1e-10)
    elif rhs.acclogp=='unary':
        logp = lhs.acclogp + np.log(parse_probs[mi,hi]+1e-10) + np.log(rel_probs[mi,hi,:][lb]+1e-10)
    else:
        logp = lhs.acclogp + rhs.acclogp + np.log(parse_probs[mi,hi]+1e-10) + np.log(rel_probs[mi,hi,:][lb]+1e-10)
    
    ## rescoring
    if rescore_config['RESCORE']:
        newlogp = bert_rescore(logp, node, head, l_tail, r_tail, rescore_matrix, rescore_config)
    else: newlogp = logp
    print('newlogp: '+str(newlogp))

    heapq.heappush(priq, [-newlogp, kl, kr, b, lb, head, lspan, rspan])
    ##(-las_logp,u,k1,k2,Vocab.ROOT)
    
    #else:
        #rhs = derivations[rspan][kr]
        #heapq.heappush(priq, [-rhs.acclogp, kl, kr, 0, 0, lspan, rspan, is_root])

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## (C) Rescoring Function
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def bert_rescore(logp, node, head_node, l_tail, r_tail, rescore_matrix, rescore_config):
    #logp = biaffine
    alpha, beta, RESCORE = rescore_config['alpha'], rescore_config['beta'], rescore_config['RESCORE']
    ## md,hd are both verbs
    print('node: ' + str(node))
    print('head: '+str(head_node)+' ltail: '+str(l_tail)+' rtail: '+str(r_tail))
    #print(hlr_forest_d[node])
    ## '<--' or '-->'
    tail_node = l_tail if head_node==r_tail else r_tail

    if rescore_matrix[tail_node-1] is not None: # parent node is verb
        bert_score = rescore_matrix[tail_node-1][head_node-1]
        return logp + beta + alpha*np.log(bert_score)
    
    else:
        return logp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', help='dir path for parsed data (.plk)')
    parser.add_argument('--rescore_cfg', help='file path for rescore config (.cfg)')
    parser.add_argument('--rescore', default=False, type=str, help='rescore on cube pruning or not')
    parser.add_argument('--test', default=False, type=str, help='test or not')
    parser.add_argument('--K', type=int, help='K for Kbest')
    parser.add_argument('--alpha', type=float, help='alpha weight for scoring function')
    parser.add_argument('--beta', type=float, help='beta bias for scoring function')
    args = parser.parse_args()

    pkl_dir = args.pkl_dir
    rescore_cfg = args.rescore_cfg
    test = args.test
    rescore_config = {'alpha': args.alpha,
                    'beta': args.beta,
                    'RESCORE': args.rescore}

    ## pack pkl / currently separated for dev purposes
    with open(os.path.join(pkl_dir,'forests.pkl'), 'rb') as p1:
        all_forests = pkl.load(p1)
    with open(os.path.join(pkl_dir,'parse_probs.pkl'), 'rb') as p2:
        all_parse_probs = pkl.load(p2)
    with open(os.path.join(pkl_dir,'rel_probs.pkl'), 'rb') as p3:
        all_rel_probs = pkl.load(p3)
    with open(os.path.join(pkl_dir,'sents.pkl'), 'rb') as p4:
        all_sents = pkl.load(p4)
    with open(os.path.join(pkl_dir,'tags.pkl'), 'rb') as p5:
        all_tags = pkl.load(p5)

    cnt=0

    print('mode')
    print(test)

    ## for evaluation
    if test=='False':
        print('loading model')
        if not os.path.exists(rescore_cfg):
            sys.exit('rescore.cfg not found')
        cfg = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
        cfg.read(rescore_cfg)
        remodel = RescoreModel(cfg)

        for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
            if cnt==2:
                break
            rescore_matrix = remodel.head_prediction(sent, tags)
            #verb_nodes = [bool(tag[0] == 'V') for tag in tags]
            print(len(forest['hyperedges']))
            main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, args.K)
            cnt+=1
        
        print('hoge')

    ## for testing purpose using an example from devset
    else:
        print('hogehoge')
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

            ## json
            forest_out = 'test/test_forest.json'
            with open(forest_out, 'w') as f:
                json.dump(forest, f)

            main_loop(forest, parse_probs, rel_probs, rescore_matrix, rescore_config, args.K)
            cnt+=1

            print(sent)
            print(tags)
            #print(parse_probs)
            #print(rel_probs.shape)

'''legacy
def backward_star(node, forest, b):
    ## a list of candidate/incoming hyperedges for an input head node

    incoming_edges = []
    head, lmost, rmost = node

    for he in forest["hyperedges"]:
        if (he['head'] == head) and (he['head_span'] == [lmost, rmost]):
            ##neglogp
            incoming_edges.append([np.log(float(he['prob']))*(-1),he['name']])

    return incoming_edges

def parse_score(forest_d, length):
    ##init
    key = [l for l in range(length)]
    val = [0]*length
    parse_score_d = dict()
    for k in key:
        parse_score_d[k] = val

    for node_id in forest_d:
        sp = node_id.split('_')
        head = int(sp[0])
        edge = list(map(int,sp[1:3]))
        tail = edge[0] if head==edge[1] else edge[1]
        parse_score_d[head][tail] = forest_d[node_id]['prob']

    return parse_score_d
'''

