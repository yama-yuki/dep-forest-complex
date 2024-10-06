'''
lib
'''
import json
import os, sys, heapq, copy
import time
import pickle as pkl
import numpy as np
from pprint import pprint
from collections import defaultdict

class Vocab:
    PAD=0
    ROOT=1
    UNK=2

class Hypo:
    ## orginal eisner hypothesis
    def __init__(self, logp, edges, u, num_roots):
        self.logp = logp
        self.edges = edges
        self.u = u
        self.num_roots = num_roots

    def __str__(self):
        pass

def cube_pruning(s, t, kk, memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, forest, length):
    if s == 0 and kk[0] == '<-': ## artificial root can't be governed
        return

    key = (s,t) + kk
    hd, md = (s,t) if kk[0] == '->' else (t,s)

    new_uas = np.log(parse_probs[md,hd]+1e-10)
    new_las = np.log(rel_probs[md,hd,:]+1e-10)

    if kk[1] == 0:
        u_range = range(s,t)
        u_inc = 1
        ll, rr = ('->',1), ('<-',1)
    elif kk[1] == 1 and kk[0] == '<-':
        u_range = range(s,t)
        u_inc = 0
        ll, rr = ('<-',1), ('<-',0)
    else:
        u_range = range(s+1,t+1)
        u_inc = 0
        ll, rr = ('->',0), ('->',1)
    #print('cube_pruning:', key, ll, rr)

    ## initialize priority queue
    priq = []
    visited = set() ## each item is (split_u, k1, k2)

    #print('init')
    for u in u_range:
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, 0, 0, new_uas, new_las, s==0)

    ## actual cube pruning
    nbest = []
    #print('cube pruning')
    while len(priq) > 0:
        ### obtain the current best
        neglogp, u, k1, k2, li = heapq.heappop(priq) ## return minimum = -maximum
        logp = -neglogp
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        edges = memory[lhs][k1].edges | memory[rhs][k2].edges

        #print('lhs ',str(memory[lhs][k1].edges))
        #print('rhs ',str(memory[rhs][k2].edges))
        num_roots = memory[lhs][k1].num_roots + memory[rhs][k2].num_roots
        if li is not None:
            edges.add((md,hd,li))
            num_roots += (s == 0)
        ### check if violates
        is_violate = (num_roots > 1)
        j = -1

        logp = nbest_rescore_function(md, hd, logp, rescore_configs)

        for i, hyp in enumerate(nbest):
            #### hypotheses with same edges should have same logp
            if is_violate or hyp.edges == edges: ##or \
                    ##(i == 0 and hyp.logp - logp >= 10.0):
                is_violate = True
                break
            if hyp.logp < logp:
                j = i
                break

        ### insert
        if not is_violate :
            new_hyp = Hypo(logp, edges, u, num_roots)

            ## hypo edges
            if kk[1]==1:
                if new_hyp.edges not in forest['hypotheses']:
                    forest['hypotheses'].append(new_hyp)
                    #print(new_hyp.edges)

            if j == -1:
                nbest.append(new_hyp)
            else:
                nbest.insert(j, new_hyp)
        if len(nbest) >= NBEST:
            break
        ### append new to priq
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1+1, k2, new_uas, new_las, s==0)
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1, k2+1, new_uas, new_las, s==0)

    memory[key] = nbest[:NBEST]

def nbest_rescore_function(md, hd, logp, rescore_configs):
    if rescore_configs['RESCORE']=='inside':
        #if rescore_configs['rescores'][md][1] is not None and rescore_configs['rescores'][hd][1] is not None: # verb-verb condition: md&hd are both verbs
        if rescore_configs['rescores'][md][1] is not None: ## verb-any condition: md is verb
            alpha = 0.1
            beta = 0.1
            logp = logp + alpha + beta*np.log(rescore_configs['rescores'][md][1][hd-1])
    return logp

def cube_next(lhs_list, rhs_list, visited, priq,
        is_making_incomplete, u, k1, k2, new_uas, new_las, is_s_0 = False):
    if len(lhs_list) <= k1 or len(rhs_list) <= k2 or \
            (u, k1, k2) in visited:
        return
    
    ## visited combination
    #print('add to visited '+str((u,k1,k2)))
    visited.add((u,k1,k2))
    #print(visited)
    ## u: span boundary
    #print('u='+str(u))
    #print('lhs:'+str(lhs_list[k1].edges))
    #for lhs in lhs_list:
        #print(lhs.edges)
    #print('rhs:'+str(rhs_list[k2].edges))
    #for rhs in rhs_list:
        #print(rhs.edges)

    uas_logp = lhs_list[k1].logp + rhs_list[k2].logp
    if is_making_incomplete: # making incomplete hypothesis, adding an edge
        uas_logp += new_uas
        if is_s_0: # s == 0 and is making ('->', 0), must have ROOT relation
            las_logp = uas_logp + new_las[Vocab.ROOT]
            heapq.heappush(priq, (-las_logp,u,k1,k2,Vocab.ROOT))
        else:
            for i, logp in enumerate(new_las):
                if i not in (Vocab.PAD, Vocab.ROOT, Vocab.UNK):
                    las_logp = uas_logp + logp
                    heapq.heappush(priq, (-las_logp,u,k1,k2,i))
    else:
        heapq.heappush(priq, (-uas_logp,u,k1,k2,None))


'''
eisner_dp_nbest: returns nbest trees
eisner_dp_forest: returns binarized dependency forest
'''

def eisner_dp_nbest(length, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA):
    #st_time = time.time()
    forest = {'hypotheses': [], 'hyperedge_ids': [], 'token_ids': [i for i in range(int(length+1))]}

    rescore_configs = {'ALPHA':ALPHA,
                       'RESCORE':RESCORE,
                       'rescores':rescores
                       }

    '''init
    memory:
    defaultdict(<class 'list'>, {(0, 0, '->', 0): [0], (0, 0, '->', 1): [0], (0, 0, '<-', 0): [0], (0, 0, '<-', 1): [0], (1, 1, '->', 0): [0], (1, 1, '->', 1): [0], (1, 1, '<-', 0): [0], (1, 1, '<-', 1): [0], (2, 2, '->', 0): [0], (2, 2, '->', 1): [0], (2, 2, '<-', 0): [0], (2, 2, '<-', 1): [0], (3, 3, '->', 0): [0], (3, 3, '->', 1): [0], (3, 3, '<-', 0): [0], (3, 3, '<-', 1): [0], (4, 4, '->', 0): [0], (4, 4, '->', 1): [0], (4, 4, '<-', 0): [0], (4, 4, '<-', 1): [0], (5, 5, '->', 0): [0], (5, 5, '->', 1): [0], (5, 5, '<-', 0): [0], (5, 5, '<-', 1): [0]})
    '''
    memory = defaultdict(list)
    for i in range(0, length+1): ##token_len
        for d in ('->', '<-'): ##direction
            for c in range(2): ##completeness (0:incomplete, 1: complete)
                memory[(i,i,d,c)].append(Hypo(0.0, set(), None, 0))

    for t in range(1, length+1):
        for s in range(t-1, -1, -1):
            cube_pruning(s, t, ('<-',0), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, forest, length)
            cube_pruning(s, t, ('->',0), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, forest, length)
            cube_pruning(s, t, ('<-',1), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, forest, length)
            cube_pruning(s, t, ('->',1), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, forest, length)

    nbest = []
    for hyp in memory[(0,length,'->',1)]:
        nbest.append([])
        for mi,hi,lb in hyp.edges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((prb,mi,hi,lb))

    '''ids
    mi: child idx
    hi: head idx
    lb: relation label
    '''

    return nbest

def eisner_dp_forest(length, parse_probs, rel_probs, rel_vocab, NBEST):
    hypos = {'hypotheses': []}

    ## constants

    rescore_configs = {'ALPHA':None,
                       'RESCORE':False,
                       'rescores':None
                       }

    memory = defaultdict(list)
    for i in range(0, length+1): ##token_len
        for d in ('->', '<-'): ##direction
            for c in range(2): ##completeness (0:incomplete, 1: complete)
                memory[(i,i,d,c)].append(Hypo(0.0, set(), None, 0))

    for t in range(1, length+1):
        for s in range(t-1, -1, -1):
            cube_pruning(s, t, ('<-',0), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, hypos, length)
            cube_pruning(s, t, ('->',0), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, hypos, length)
            cube_pruning(s, t, ('<-',1), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, hypos, length)
            cube_pruning(s, t, ('->',1), memory, parse_probs, rel_probs, rel_vocab, rescore_configs, NBEST, hypos, length)

    '''original kbest eisner
    ## output nbest of memory[(0,length,'->',1)]

    for hyp in memory[(0,length,'->',1)]:
        print hyp.edges, hyp.logp, hyp.num_roots
    print('Length %d, time %f' %(length, time.time()-st_time))
    return [list(hyp.edges) for hyp in memory[(0,length,'->',1)]] # return edges containing (mi,hi,lb)
    '''

    return hypos

if __name__ == '__main__':
    ## do some unit test
    sys.path.append('/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest')

    pkl_dir = '/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl_complete'
    with open(os.path.join(pkl_dir,'parse_probs.pkl'), 'rb') as p2:
        all_parse_probs = pkl.load(p2)
    with open(os.path.join(pkl_dir,'rel_probs.pkl'), 'rb') as p3:
        all_rel_probs = pkl.load(p3)
    parse_probs=all_parse_probs[0]
    rel_probs=all_rel_probs[0]

    rescores = RESCORE = ALPHA = None

    length = 9
    NBEST = 128
    rel_vocab=[0]*10**3
    forest = eisner_dp_forest(length, parse_probs, rel_probs, rel_vocab, NBEST)

    print(len(forest['hyperedges']))

    ##json
    out_dir = '/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/'
    forest_out = 'test_forest.json'
    with open(out_dir+forest_out, 'w') as f:
        json.dump(forest, f)

