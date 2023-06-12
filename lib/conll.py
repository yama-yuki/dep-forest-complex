'''
module for operation on trees/forests
'''

import heapq
import os
import pickle as pkl
from collections import defaultdict
from pprint import pprint

class Tree:
    '''
    tree structure to organize edges for output
    edge: (head, tail)
    '''
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

def final_1best(length, derivations, parse_probs, rel_probs):
    '''
    output 1best set of edges from derivation chart
    '''
    print('final_derivation')
    final_kbests = derivations[(0,-1,length-1)]
    print(final_kbests)
    
    for j,kbest in enumerate(final_kbests):
        print(kbest)
        print(kbest[-1].depedges)
    best_hyp = final_kbests[0][-1]

    print('best_hyp')
    print(best_hyp)
    print(best_hyp.node_names)

    edges = []
    for hd,md,lb,deprel in best_hyp.depedges_l:
        prb = parse_probs[md,hd] * rel_probs[md,hd,:][lb]
        assert prb > 0.0
        edges.append((md,hd,lb,deprel,prb))#prb

    best_tree = sorted(edges, key=lambda x: x[0]) 
    
    #kbests = sorted(finals, key=lambda x: x.acclogp, reverse=True)
    #kbests = sorted(finals, key=lambda x: x.acclogp)

    return best_tree

def to_conllu(out_path, best_tree, sent, tags):

    temp_d = defaultdict()
    for edge in best_tree:
        tail, head, label, deprel, prb = edge
        temp_d[tail] = (head,deprel)
        print(edge)
    
    keys = list(sorted(temp_d.keys()))
    conll_lines = ['\t'.join([str(tail),sent[tail-1],sent[tail-1].lower(),tags[tail-1],'_','_','_','_',str(temp_d[tail][0]),str(temp_d[tail][1])]) for tail in keys]

    with open(out_path, mode='a', encoding='utf-8') as o:
        o.write('# '+' '.join(sent)+'\n')
        for line in conll_lines:
            o.write(line+'\n')
        o.write('\n')

def pkl_loader(pkl_dir):
    #pkl_dir = '/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl/k4'

    all_forests = []
    all_parse_probs, all_rel_probs = [], []
    all_sents, all_tags = [], []

    for n in range(1,11): ## id of buckets in parser
        with open(os.path.join(pkl_dir,str(n)+'forests.pkl'), 'rb') as p1:
            forests = pkl.load(p1)
            all_forests.extend(forests)
        with open(os.path.join(pkl_dir,str(n)+'parse_probs.pkl'), 'rb') as p2:
            parse_probs = pkl.load(p2)
            all_parse_probs.extend(parse_probs)
        with open(os.path.join(pkl_dir,str(n)+'rel_probs.pkl'), 'rb') as p3:
            rel_probs = pkl.load(p3)
            all_rel_probs.extend(rel_probs)
        with open(os.path.join(pkl_dir,str(n)+'sents.pkl'), 'rb') as p4:
            sents = pkl.load(p4)
            all_sents.extend(sents)
        with open(os.path.join(pkl_dir,str(n)+'tags.pkl'), 'rb') as p5:
            tags = pkl.load(p5)
            all_tags.extend(tags)
        print('Number of Parsed Forests: '+str(len(sents)))

    print('Total: '+str(len(all_sents)))

    return all_forests, all_parse_probs, all_rel_probs, all_sents, all_tags

