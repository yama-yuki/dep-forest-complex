from collections import defaultdict
from tqdm import tqdm

class Tree:
    ## construct dependency tree from a set of edges
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

class BinHyperedge2:
    def __init__(self, X, A, B, Xspan, Aspan, Bspan, label_id, X_id, A_id, B_id, parse_prob, rel_prob):
        self.X = X
        self.A = A
        self.B = B
        self.Xspan = Xspan
        self.Aspan = Aspan
        self.Bspan = Bspan
        self.label_id = label_id
        self.X_id = X_id
        self.A_id = A_id
        self.B_id = B_id
        self.parse_prob = str(parse_prob)
        self.rel_prob = str(rel_prob)
    
    def as_dict(self):
        return {'X':self.X,
                'A':self.A,
                'B':self.B,
                'Xspan':self.Xspan,
                'Aspan':self.Aspan,
                'Bspan':self.Bspan,
                'label_id':self.label_id,
                'X_id':self.X_id,
                'A_id':self.A_id,
                'B_id':self.B_id,
                'parse_prob':self.parse_prob,
                'rel_prob':self.rel_prob}

class DepHeadBinarizer:
    '''Head Binarization
    rules:
    x -> left_tail x'
    x -> x' right_tail
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

        if not l_tails and not r_tails: ##terminal
            self.visited.add(node)
            return
        
        if r_tails and l_tails:
            governed = []
            dfs_for_span(tree, node, governed)
            
            for rt in r_tails:
                ## x -> x, rt
                rt_governed = []
                dfs_for_span(tree, rt, rt_governed)

                tails = [node,rt]
                edge = self._make_edge(tree, node, rt, tails, governed, rt_governed, 'rt')
                self.actions.append(edge)
                self.cfg_conversion(tree, rt)
                governed=[gov for gov in governed if gov not in set(rt_governed)|{rt}]

            for lt in l_tails:
                ## x -> lt, x

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
                #print('rt')
                #print(rt)
                #print(rt_governed)
                tails = [node,rt]
                edge = self._make_edge(tree, node, rt, tails, rt_governed, rt_governed, 'rt')
                self.actions.append(edge)
                self.cfg_conversion(tree, rt)

        elif l_tails:
            for lt in l_tails:
                ## x -> lt, x
                lt_governed = []
                dfs_for_span(tree, lt, lt_governed)
                #print('lt')
                #print(lt)
                #print(lt_governed)
                tails = [lt,node]
                edge = self._make_edge(tree, node, lt, tails, lt_governed, lt_governed, 'lt')
                self.actions.append(edge)
                self.cfg_conversion(tree, lt)

        self.visited.add(node)

    def _make_edge(self, tree, node, tail, tails, governed1, governed2, tail_side):
        label = tree.label_d[tail,node]
        head_gov_set = set(governed1+tails)
        tail_gov_set = set(governed2)|{tail}
 
        a = min(head_gov_set)-1
        c = max(tail_gov_set) if tail_side=='lt' else min(tail_gov_set)-1
        b = max(head_gov_set)
        span_ranges = [a, c, b]

        node_name = '_'.join(map(str,[node]+tails+span_ranges+[label]))
        edge = (node,tails,span_ranges,node_name) #(1, [1, 5], [0, 4, 9], '1_1_5_0_4_9_17')
        #print(edge)
        return edge
    
    def __str__(self):
        pass  

def dfs_for_span(tree, node, governed):
    governed.append(node)
    for tail in tree.relation[node]:
        if tail not in governed:
            dfs_for_span(tree, tail, governed)

def binarize_actions(edges):
    edges = sorted(list(edges), key=lambda x: x[1])
    d_tails_labels = defaultdict(dict) ## {head:[(tail,label), ()], head:...}
    for edge in edges: ##(9, 6, 1)
        d_tails_labels[(edge[1],edge[0])] = edge[2]
    tree = Tree(edges)
    top = tree.find_top()

    ## binarize dep tree (to resolve spurious amb.)
    db = DepHeadBinarizer(tree,top)

    return db.actions, d_tails_labels


def form_cfg_hyperedges(hypo, parse_probs, rel_probs, rel_vocab):
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
    edges = sorted(list(hypo.edges), key=lambda x: x[1])
    d_tails_labels = defaultdict(dict) ## {head:[(tail,label), ()], head:...}
    for edge in edges: ##(9, 6, 1)
        d_tails_labels[edge[1]].update([(edge[0],edge[2])])
    tree = Tree(edges)

    if len(tree.relation[0])>1:
        #print('invalid_tree')
        return
    top = tree.find_top()

    ## binarize dep tree (to resolve spurious amb.)
    #actions,visited = [],set()
    #cfg_conversion(tree, top, actions, visited)
    db = DepHeadBinarizer(tree,top) 
    #print(db.actions)
    ## create a set of hyperedges
    hyperedges = set()
    Xspan2tails = defaultdict(list)

    return hyperedges

