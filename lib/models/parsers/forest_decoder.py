import os, sys
import json
import heapq
import numpy as np
import pickle as pkl
from collections import Counter, defaultdict

from pprint import pprint
from tqdm import tqdm

from rescore_module.rescore_main import RescoreModel

class Tree:
    def __init__(self, edges):
        self.relation = defaultdict(list)
        self.heads = set()
        self.tails = set()
        for (head, tail) in edges:
            self.relation[head].append(tail)
            self.heads.add(head)
            self.tails.add(tail)
        self._sort_rel()
    
    def _sort_rel(self):
        for head in self.relation:
            self.relation[head] = sorted(self.relation[head])
    
    def find_top(self):
        for head in self.heads:
            if head not in self.tails:
                return head

## derivation D
class HypoD:
    ## Tree
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

def backward_star(node, forest, b):
    ## a list of candidate/incoming hyperedges for an input head node

    incoming_edges = []
    head, lmost, rmost = node

    for he in forest["hyperedges"]:
        if (he['head'] == head) and (he['head_span'] == [lmost, rmost]):
            ##neglogp
            incoming_edges.append([np.log(float(he['prob']))*(-1),he['name']])

    return incoming_edges

def topological_sort(nodes):
    ## sort nodes based on span length (rising order)
    #9_3_9_0_4_9
    #head_lt_rt_lgov_bound_rgov
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
            nodes_hlr.append((h,l,r)) ## head,lgov,rgov

    print('topological sort')
    print(nodes_hlr)

    return nodes_hlr

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

def main_loop(forest, parse_probs, rel_probs, rescores, rescore_config, K):
    length = len(forest['node_ids'])
    print('words: '+str(length-1))
    print('with root: '+str(length))
    print('total nodes: '+str(len(forest['nodes'])))

    nodes_hlrlbr, nodes_hlr, forest_d, hlr_forest_d = load_forest(forest)
    #parse_score_d = parse_score(forest_d, length)

    '''
    nodes_hlrlbr: head, ltail, rtail, lgov, boundary, rgov
    nodes_hlr: head, lmost, rmost
    forest_d: node_id -> hyperedge
    hlr_forest_d: span_range(head, lmost, rmost) -> [node_ids]
    '''

    ## create a chart of derivations
    derivations = defaultdict(list)
    ## (1,0,1), ... (9,0,9), (1,1,2)
    
    ## initialize with unary hyperedges
    for x in range(length): ## head loop
        for alpha in range(length): ## left tail loop
            for beta in range(length): ## right tail loop
                #if alpha < x <= beta: 
                if beta-alpha==1 and x==beta:
                    derivations[(x,alpha,beta)].append(HypoD(x, 'unary', set(), set(), -1, 0))

    print(derivations)

    ## main loop
    for i,node in enumerate(nodes_hlr):
        #nodes: [(2, 0, 2), (3, 2, 4), (5, 4, 6)] (head, lgov, rgov)
        print('start_loop')
        print(node)
        select_k(i, node, derivations, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config, K)

    ## the goal is to get to the root he(0,0,length-1) with all node_ids in it
    print('final_derivation')
    #final = derivations[(0,-1,length-1)]
    #print(final)

    finals = []
    for i in range(1,length):
        print(i)
        final = derivations[(i,0,length-1)]
        finals.extend(final)
        print(final)
        if final:
            for j,f in enumerate(final):
                print(final[j].depedges)
    
    kbests = sorted(finals, key=lambda x: x.acclogp, reverse=True)
    #kbests = sorted(finals, key=lambda x: x.acclogp)

    tree = Tree(kbests[0].depedges)
    top = tree.find_top()
    kbests[0].depedges.add((0, top))
    print(kbests[0].acclogp)
    print(kbests[0].depedges)
    print(kbests[1].acclogp)
    print(kbests[1].depedges)

    nbest = []
    for hyp in kbests:
        nbest.append([])
        for hi,mi,lb in hyp.hyperedges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((hi,mi,lb,prb))
    
    pprint(sorted(nbest[0], key=lambda x: x[1]))

def select_k(i, node, derivations, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config, K):
    # TODO: control K 10/100/1000
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
        cube_next(derivations, node, 0, 0, b, lb, head, lspan, rspan, visited, priq, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config)

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
        cube_next(derivations, node, kl+1, kr, b, lb, head, lspan, rspan, visited, priq, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config)
        cube_next(derivations, node, kl, kr+1, b, lb, head, lspan, rspan, visited, priq, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config)
    ## sort buffer to D(v)
    #best_K_buffer = sorted(best_K_buffer, key=lambda x: x.acclogp, reverse=True)
    derivations[node] = best_K_buffer[:K]
    #print(visited)

def cube_next(derivations, node, kl, kr, b, lb, head, lspan, rspan, visited, priq, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config):
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
    print(l_tail, r_tail)
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
    newlogp = bert_rescore(logp, node, head, l_tail, r_tail, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config)
    heapq.heappush(priq, [-newlogp, kl, kr, b, lb, head, lspan, rspan])
    ##(-las_logp,u,k1,k2,Vocab.ROOT)
    
    #else:
        #rhs = derivations[rspan][kr]
        #heapq.heappush(priq, [-rhs.acclogp, kl, kr, 0, 0, lspan, rspan, is_root])

def bert_rescore(logp, node, head_node, l_tail, r_tail, forest_d, hlr_forest_d, parse_probs, rel_probs, rescores, rescore_config):
    #logp = biaffine
    alpha, beta, RESCORE = rescore_config['alpha'], rescore_config['beta'], rescore_config['RESCORE']
    ## md,hd are both verbs
    print('node: ' + str(node))
    print('head: '+str(head_node)+' ltail: '+str(l_tail)+' rtail: '+str(r_tail))
    #print(hlr_forest_d[node])

    tail_node = l_tail  if head_node==r_tail else r_tail

    if RESCORE:
        if rescores[tail_node-1] is None:
            newlogp = logp
        else:
            bert_score = rescores[tail_node-1][head_node-1]
            newlogp = logp + alpha + beta*np.log(bert_score)
    else:
        newlogp = logp
    print('newlogp: '+str(newlogp))

    return newlogp

if __name__ == '__main__':
    ## do some unit test

    cur_dir = '/home/is/yuki-yama/work/d3/dep-forest/biaffine_forest'
    with open(os.path.join(cur_dir,'pkl/forests.pkl'), 'rb') as p1:
        all_forests = pkl.load(p1)
    with open(os.path.join(cur_dir,'pkl/parse_probs.pkl'), 'rb') as p2:
        all_parse_probs = pkl.load(p2)
    with open(os.path.join(cur_dir,'pkl/rel_probs.pkl'), 'rb') as p3:
        all_rel_probs = pkl.load(p3)
    with open(os.path.join(cur_dir,'pkl/sents.pkl'), 'rb') as p4:
        all_sents = pkl.load(p4)
    with open(os.path.join(cur_dir,'pkl/tags.pkl'), 'rb') as p5:
        all_tags = pkl.load(p5)

    rescore_config = {'alpha': 10.0,
                      'beta': 10.0,
                      'RESCORE': True}

    print('loading model')
    remodel = RescoreModel('models')
    cnt=0
    for forest,parse_probs,rel_probs,sent,tags in tqdm(zip(all_forests,all_parse_probs,all_rel_probs,all_sents,all_tags)):
        if cnt==1:
            break
        rescores = remodel.head_prediction(sent, tags)
        '''
        rescores = [None, None, None, None, np.array([4.8442665e-08, 4.3052935e-08, 4.3104702e-08, 4.2075676e-08,
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
        #print(rescores)
        '''

        verb_nodes = [bool(tag[0] == 'V') for tag in tags]
        #print(forest)
        print(len(forest['hyperedges']))

        ##json
        forest_out = 'for.json'
        with open(forest_out, 'w') as f:
            json.dump(forest, f)

        #sys.exit()
        print(rescores)
        main_loop(forest, parse_probs, rel_probs, rescores, rescore_config, 3)
        cnt+=1

        print(sent)
        print(tags)
        #print(parse_probs)
        #print(rel_probs.shape)

