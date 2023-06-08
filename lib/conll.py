import heapq
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
    print(temp_d)
    print(keys)
    conll_lines = ['\t'.join([str(tail),sent[tail-1],sent[tail-1].lower(),tags[tail-1],'_','_','_','_',str(temp_d[tail][0]),str(temp_d[tail][1])]) for tail in keys]
    print(conll_lines)
    with open(out_path, mode='w', encoding='utf-8') as o:
        o.write('# '+' '.join(sent)+'\n')
        for line in conll_lines:
            o.write(line+'\n')
        o.write('\n')

