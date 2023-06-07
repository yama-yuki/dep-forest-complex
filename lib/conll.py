import conllu

from collections import defaultdict
from pprint import pprint

class Tree:
    '''
    tree structure to organize edges for output
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

def to_conllu(length, derivations, parse_probs, rel_probs):
    finals = []
    for i in range(1,length):
        print(i)
        final = derivations[(i,0,length-1)]
        finals.extend(final)
        if final:
            for j,f in enumerate(final):
                print(f.depedges)
    
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
            nbest[-1].append((hi,mi,lb,))#prb
    
    pprint(sorted(nbest[0], key=lambda x: x[1]))

