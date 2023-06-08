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
    kbests[1].depedges.add((0, top))
    print(kbests[1].acclogp)
    print(kbests[1].depedges)

    nbest = []
    for hyp in kbests:
        nbest.append([])
        for hi,mi,lb,deprel in hyp.hyperedges:
            prb = parse_probs[mi,hi] * rel_probs[mi,hi,lb]
            assert prb > 0.0
            nbest[-1].append((hi,mi,lb,deprel))#prb
    
    best_tree = sorted(nbest[0], key=lambda x: x[1]) 

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

