'''
lib
'''
import os, sys, heapq, copy
import time
import numpy as np
from pprint import pprint
from collections import defaultdict
##from vocab import Vocab
class Vocab:
    PAD=0
    ROOT=1
    UNK=2

import json
sys.path.append('/home/is/yuki-yama/work/d3/dep-forest/biaffine_forest')#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#NBEST = 3
#RESCORE = True

class Hypo:
    ## orginal eisner hypothesis
    def __init__(self, logp, edges, u, num_roots):
        self.logp = logp
        self.edges = edges
        self.u = u
        self.num_roots = num_roots

    def __str__(self):
        pass

class Tree:
    ## construct dep tree from a set of edges
    def __init__(self, edges):
        self.relation = defaultdict(list)
        self.label_d = defaultdict()
        self.heads = set()
        self.tails = set()
        for (tail, head, label) in edges:
            self.relation[head].append(tail)
            self.label_d[(tail,head)] = label
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

class BinHyperedge:
    ## Binarized Hyperedge Structure
    def __init__(self, he_name, head, left_tail, right_tail, label_id, label, prob, tri_span):
        '''
        args:
            head: head node
            tail: tail node
            label: deprel
            prob: parse_prob
            span_ids: (head_span, tail_span)
        '''
        self.he_name = he_name
        self.head = head
        self.left_tail = left_tail
        self.right_tail = right_tail
        self.label_id = label_id
        self.label = label
        self.prob = prob
        self.tri_span = tri_span
    
    def as_dict(self):
        return {'name':self.he_name,
                'head':self.head,
                'left_tail':self.left_tail,
                'right_tail':self.right_tail,
                'head_span':[self.tri_span[0],self.tri_span[2]],
                'left_span':[self.tri_span[0],self.tri_span[1]],
                'right_span':[self.tri_span[1],self.tri_span[2]],
                'label_id':self.label_id,
                'label':self.label,
                'prob':str(self.prob)}

class DepHeadBinarizer:
    '''Head Binarization
    rules:
    x -> left_tail x
    x -> x right_tail
    '''

    def __init__(self, tree, top):
        self.tree = tree
        self.top = top
        self.actions = []
        self.visited = set()
        self.cfg_conversion(self.tree, self.top)
    
    def cfg_conversion(self, tree, node):
        '''
        tree.relation: {head: [tails], head: [tails], ...}
        '''
        r_tails = sorted([tail for tail in tree.relation[node] if tail>node],reverse=True)
        l_tails = sorted([tail for tail in tree.relation[node] if tail<node])
        print('---')
        print(node)
        print(r_tails)
        print(l_tails)
        print('---')

        if not l_tails and not r_tails: #terminal
            self.visited.add(node)
            #print(sorted(list(self.visited)))
            return
        
        if r_tails and l_tails:
            governed = []
            dfs_for_span(tree, node, governed)
            
            for rt in r_tails:
                ## x -> x, rt
                print('rt')
                print(rt)
                print(sorted(governed))

                rt_governed = []
                dfs_for_span(tree, rt, rt_governed)

                tails = [node,rt]
                edge = self._make_edge(tree, node, rt, tails, governed, rt_governed, 'rt')
                self.actions.append(edge)
                self.cfg_conversion(tree, rt)
                governed=[gov for gov in governed if gov not in set(rt_governed)|{rt}]

            for lt in l_tails:
                ## x -> lt, x
                print('lt')
                print(lt)
                print(sorted(governed))

                lt_governed = []
                dfs_for_span(tree, lt, lt_governed)
                tails = [lt,node]
                edge = self._make_edge(tree, node, lt, tails, governed, lt_governed, 'lt')
                self.actions.append(edge)
                self.cfg_conversion(tree, lt)
                governed=[gov for gov in governed if gov not in set(lt_governed)|{lt}]

        elif r_tails:
            for rt in r_tails:
                ## x -> x, rt
                rt_governed = []
                dfs_for_span(tree, rt, rt_governed)
                print('rt')
                print(rt)
                print(rt_governed)
                tails = [node,rt]
                edge = self._make_edge(tree, node, rt, tails, rt_governed, rt_governed, 'rt')
                self.actions.append(edge)
                self.cfg_conversion(tree, rt)

        elif l_tails:
            for lt in l_tails:
                ## x -> lt, x
                lt_governed = []
                dfs_for_span(tree, lt, lt_governed)
                print('lt')
                print(lt)
                print(lt_governed)
                tails = [lt,node]
                edge = self._make_edge(tree, node, lt, tails, lt_governed, lt_governed, 'lt')
                self.actions.append(edge)
                self.cfg_conversion(tree, lt)

        self.visited.add(node)
        #print(self.visited)          

    def _make_edge(self, tree, node, tail, tails, governed1, governed2, mode):
        label = tree.label_d[tail,node]
        head_gov_set = set(governed1+tails)
        tail_gov_set = set(governed2)|{tail}
 
        a = min(head_gov_set)-1
        c = max(tail_gov_set) if mode=='lt' else min(tail_gov_set)-1
        b = max(head_gov_set)
        tri_span = [a, c, b]

        node_name = '_'.join(map(str,[node]+tails+tri_span+[label]))
        edge = (node,tails,tri_span,node_name) #(1, [1, 5], [0, 4, 9], '1_1_5_0_4_9')
        print(edge)
        return edge
    
    def __str__(self):
        pass  

def dfs_for_span(tree, node, governed):
    governed.append(node)
    for tail in tree.relation[node]:
        if tail not in governed:
            dfs_for_span(tree, tail, governed)

def form_cfg_hyperedges(edges, parse_probs, rel_probs, rel_vocab):
    '''
    ## rewrite dep v->u to x_v->v,u cfg-style
    ## right-element-first merge to handle spurious ambiguity

    args:
        hypo: logp,edges,u,num_root
            hypo.edges: {(tail, head, label)}
        parse_probs
        rel_probs
        span_ids: defaultdict(<class 'list'>, {1: [0, 9], 2: [1, 2], 3: [2, 3], 4: [2, 4], 5: [4, 9], 6: [5, 6], 7: [6, 7], 8: [7, 8], 9: [6, 9]})

    return:
        hyperedges:
            tails, probs, labels, head, span_ids = [], [], [], head, [[],[]]
            #[self.tails, self.head, self.labels, self.probs, self.span_ids]
    '''
    
    ## make tree from edges
    #edges = sorted(list(hypo.edges), key=lambda x: x[1]) ##{(9, 6, 1), (3, 2, 1), (7, 9, 1), (6, 5, 1), (5, 4, 1), (8, 9, 1), (4, 2, 1)}
    edges = sorted(list(edges), key=lambda x: x[1])
    d_tails_labels = defaultdict(dict) ## {head:[(tail,label), ()], head:...}
    for edge in edges: ##(9, 6, 1)
        d_tails_labels[edge[1]].update([(edge[0],edge[2])])
    tree = Tree(edges)

    if len(tree.relation[0])>1:
        print('invalid_tree')
        return
    top = tree.find_top()

    ## binarize dep tree (to resolve spurious amb.)
    #actions,visited = [],set()
    #cfg_conversion(tree, top, actions, visited)
    db = DepHeadBinarizer(tree,top) 
    print(db.actions)

    ## create a set of hyperedges
    hyperedges = set()
    for action in db.actions:
        head,tails,tri_span,name = action
        left_tail,right_tail = tails
        ## compute prob
        tail = left_tail if head!=left_tail else right_tail
        label_id = d_tails_labels[head][tail]
        label = rel_vocab[label_id]
        prob = str(parse_probs[tail,head]*rel_probs[tail,head,:][label_id])
        he = BinHyperedge(name,head,left_tail,right_tail,label_id,label,prob,tri_span)
        hyperedges.add(he)

    return hyperedges

def cube_pruning(s, t, kk, memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length):
    if s == 0 and kk[0] == '<-': ## artificial root can't be governed
        return

    key = (s,t) + kk
    hd, md = (s,t) if kk[0] == '->' else (t,s)

    #new_parse_score = outside_rescore_function(md, hd, parse_probs, rescores, RESCORE, ALPHA)
    new_uas = np.log(parse_probs[md,hd]+1e-10)#new_uas = np.log(parse_probs[md,hd]+1e-10)
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

        logp = rescore_function(md, hd, logp, rescores, RESCORE, ALPHA)

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
            #print(hd, md)
            #print(edges)
            #if edges == {(9, 2, 3), (3, 2, 19), (6, 8, 20), (8, 5, 28), (7, 8, 8), (5, 2, 18), (2, 1, 21), (4, 5, 11)}:
                #print('-----')
                #print(kk[1])
            '''
            flag=True
            for edge in edges:
                if edge[0]==2 and edge[1]!=0:
                    flag=False
            if flag==True and len(edges)==9:
                print(edges)
            '''

            '''kbest merging method of creating a forest based on original paper
            ## merge hypotheses using all edges
            if len(new_hyp.edges) == length:
                print(new_hyp.edges)
                ## create hyperedge representation
                span_ids = assign_id(new_hyp)
                hyperedges = form_cfg_hyperedges(new_hyp, parse_probs, rel_probs)
                print([he.as_list() for he in hyperedges])
                for he in hyperedges:
                    if he.as_dict() not in forest['hyperedges']:
                        forest['hyperedges'].append(he.as_dict())
                        forest['nodes'].append(he.he_name)
            '''


            ## create hyperedge representation
            if kk[1]==1: ## only when is_making_complete
                #lhs_he = form_cfg_hyperedges(memory[lhs][k1].edges, parse_probs, rel_probs, rel_vocab)
                #rhs_he = form_cfg_hyperedges(memory[rhs][k2].edges, parse_probs, rel_probs, rel_vocab)
                all_he = form_cfg_hyperedges(edges, parse_probs, rel_probs, rel_vocab)
                #wo_root_he = form_cfg_hyperedges(set(filter(lambda x: x[1]!=0, edges)), parse_probs, rel_probs, rel_vocab)
                print(edges)
                #print(set(filter(lambda x: x[1]!=0, edges)))
                #print([he.as_list() for he in hyperedges])
                #for hyperedges in [lhs_he, rhs_he, all_he, wo_root_he]:
                for hyperedges in [all_he]:
                    if hyperedges is not None:
                        for he in hyperedges:
                            print(he.he_name)
                            if he.as_dict() not in forest['hyperedges']:
                                forest['hyperedges'].append(he.as_dict())
                                forest['nodes'].append(he.he_name)


            '''
            ## create hyperedge representation
            if kk[1]==1: ## only when is_making_complete
                hyperedges = form_cfg_hyperedges(new_hyp, parse_probs, rel_probs, rel_vocab)
                #print([he.as_list() for he in hyperedges])
                if hyperedges is not None:
                    for he in hyperedges:
                        if he.as_dict() not in forest['hyperedges']:
                            forest['hyperedges'].append(he.as_dict())
            '''

            '''
            ## include incomplete
            hyperedges = form_cfg_hyperedges(new_hyp, parse_probs, rel_probs, rel_vocab)
            #print([he.as_list() for he in hyperedges])
            if hyperedges is not None:
                for he in hyperedges:
                    if he.as_dict() not in forest['hyperedges']:
                        forest['hyperedges'].append(he.as_dict())
                        forest['nodes'].append(he.he_name)
            '''

            if j == -1:
                nbest.append(new_hyp)
            else:
                nbest.insert(j, new_hyp)
        if len(nbest) >= NBEST:
            break
        ### append new to priq
        #print('next')
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1+1, k2, new_uas, new_las, s==0)
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1, k2+1, new_uas, new_las, s==0)
    memory[key] = nbest[:NBEST]

def rescore_function(md, hd, logp, rescores, RESCORE, ALPHA):
    if RESCORE=='inside':
        ## verb-verb condition: md&hd are both verbs
        if rescores[md][1] is not None and rescores[hd][1] is not None:
        #if rescores[md][1] is not None: ## verb-any condition: md is verb
            #rescore = np.log(ALPHA+rescores[md][1][hd-1])
            alpha = 1.0
            beta = 1.5
            logp = logp + alpha + beta*np.log(rescores[md][1][hd-1])
            #print(logp)
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
    forest = {'hyperedges': [], 'nodes': [], 'node_ids': [i for i in range(int(length+1))]}
    
    memory = defaultdict(list)
    for i in range(0, length+1): ##token_len
        for d in ('->', '<-'): ##direction
            for c in range(2): ##completeness (0:incomplete, 1: complete)
                memory[(i,i,d,c)].append(Hypo(0.0, set(), None, 0))

    for t in range(1, length+1):
        for s in range(t-1, -1, -1):
            cube_pruning(s, t, ('<-',0), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('->',0), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('<-',1), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('->',1), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)

    nbest = []
    for hyp in memory[(0,length,'->',1)]:
        nbest.append([])
        for mi,hi,lb in hyp.edges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((prb,mi,hi,lb))

    '''
    print(len(forest['hyperedges']))
    len_file='len.out'
    with open(len_file, 'a') as f:
        f.write(str(len(forest['node_ids']))+' '+str(len(forest['hyperedges'])))
        f.write('\n')
    '''

    return nbest

def eisner_dp_forest(length, parse_probs, rel_probs, rel_vocab, NBEST):
    #st_time = time.time()
    forest = {'hyperedges': [], 'nodes': [], 'node_ids': [i for i in range(int(length+1))]}

    ##constant
    ALPHA=None
    RESCORE=False
    rescores=None

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
            cube_pruning(s, t, ('<-',0), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('->',0), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('<-',1), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)
            cube_pruning(s, t, ('->',1), memory, parse_probs, rel_probs, rel_vocab, rescores, RESCORE, NBEST, ALPHA, forest, length)

    '''original kbest eisner
    ## output nbest of memory[(0,length,'->',1)]

    for hyp in memory[(0,length,'->',1)]:
        print hyp.edges, hyp.logp, hyp.num_roots
    print('Length %d, time %f' %(length, time.time()-st_time))
    return [list(hyp.edges) for hyp in memory[(0,length,'->',1)]] # return edges containing (mi,hi,lb)
    '''

    '''ids
    mi: child idx
    hi: head idx
    lb: relation label
    '''

    nbest = []
    for hyp in memory[(0,length,'->',1)]:
        nbest.append([])
        for mi,hi,lb in hyp.edges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((prb,mi,hi,lb))

    print(len(forest['hyperedges']))
    '''
    len_file='len.out'
    with open(len_file, 'a') as f:
        f.write(str(len(forest['node_ids']))+' '+str(len(forest['hyperedges'])))
        f.write('\n')
    '''

    return forest

if __name__ == '__main__':
    ## do some unit test

    import pickle as pkl

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

    '''original
    Vocab.ROOT = 0
    parse_probs = np.arange(1, 10001, dtype=np.float32).reshape((100,100))
    parse_probs = parse_probs/np.sum(parse_probs, axis=-1, keepdims=True)
    rel_probs = np.arange(1, 300001, dtype=np.float32).reshape((100,100,30))
    rel_probs = rel_probs/np.sum(rel_probs, axis=-1, keepdims=True)
    print(parse_probs)
    print(rel_probs)
    rescores = None
    eisner_dp_nbest(99, parse_probs, rel_probs, rescores)
    print(Vocab.ROOT)
    '''

    '''
    ##(prb, mi:child, hi:head, lb:label)
    Vocab.ROOT = 0
    length = 0
    np.random.seed(0)
    parse_probs = np.arange(1, 10001, dtype=np.float32).reshape((100,100))
    rel_probs = np.arange(1, 300001, dtype=np.float32).reshape((100,100,30))
    parse_probs = np.random.permutation(np.arange(1, 101, dtype=np.float32)).reshape((10,10))
    rel_probs = np.random.permutation(np.arange(1, 301, dtype=np.float32)).reshape((10,10,3))
    parse_probs = parse_probs/np.sum(parse_probs, axis=-1, keepdims=True)
    rel_probs = rel_probs/np.sum(rel_probs, axis=-1, keepdims=True)

    rescores = RESCORE = ALPHA = None
    NBEST = 128
    nbest, forest = eisner_dp_forest(9, parse_probs, rel_probs, rescores, RESCORE, NBEST, ALPHA)

    pprint(sorted(nbest[0]))
    print(len(nbest[0]))

    print(len(forest['hyperedges']))

    ##json
    out_dir = '/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/'
    forest_out = 'test_forest.json'
    with open(out_dir+forest_out, 'w') as f:
        json.dump(forest, f)
    '''

