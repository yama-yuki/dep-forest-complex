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
    tmp = []
    heapq.heapify(tmp)
    for i in range(0,length):
        print(i)
        final_derivations = derivations[(i,-1,length-1)]
    
        if final_derivations:
            print(final_derivations)
            for j,derivation in enumerate(final_derivations):
                print(derivation)
                heapq.heappush(tmp,derivation)
                print(derivation[-1].depedges)
    
    #kbests = sorted(finals, key=lambda x: x.acclogp, reverse=True)
    #kbests = sorted(finals, key=lambda x: x.acclogp)

    '''
    tree = Tree(tmp[0].depedges)
    top = tree.find_top()
    tmp[0].depedges.add((0, top))
    print(tmp[0].acclogp)
    print(tmp[0].depedges)
    tmp[1].depedges.add((0, top))
    print(tmp[1].acclogp)
    print(tmp[1].depedges)
    '''

    kbests = []
    for hyp in tmp:
        kbests.append([])
        for hi,mi,lb,deprel in hyp.depedges_l:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            kbests[-1].append((hi,mi,lb,deprel))#prb
    
    best_tree = sorted(kbests[0], key=lambda x: x[1]) 

    return best_tree

def to_conllu(out_path, best_tree, sent, tags):

    temp_d = defaultdict()
    for edge in best_tree:
        head, tail, label, deprel = edge
        temp_d[tail] = (head,deprel)
        print(edge)
    
    keys = list(sorted(temp_d.keys()))
    conll_lines = ['\t'.join([str(tail),sent[tail-1],sent[tail-1].lower(),tags[tail-1],'_','_','_','_',str(temp_d[tail][0]),str(temp_d[tail][1])]) for tail in keys]

    with open(out_path, mode='w', encoding='utf-8') as o:
        o.write('# '+' '.join(sent)+'\n')
        for line in conll_lines:
            o.write(line+'\n')
        o.write('\n')

