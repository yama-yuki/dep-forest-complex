def old_cfg_conversion(tree, node, tmp, visited, cfg_edges):
    ## tree.relation: {head: tails, head: tails, ...}

    if (not tree.relation[node]) and tmp: #terminal: have no tails (merge with head)
        if tmp<node:
            left,right=tmp,node
        else: left,right=node,tmp
        spans = []
        cfg_edges.add((tmp,left,right,spans)) #(head, left, right, spans[(),(),()])
        return
    
    # 2
    for tail in sorted(tree.relation[node],reverse=True):
        # [5,4,3,2]
        if tail>node:
            cfg_conversion(tree, tail, node, visited, cfg_edges)
        else:
            cfg_conversion(tree, tail, tail, visited, cfg_edges)

def cfg_conversion(tree, node, actions, visited):
    ## tree.relation: {head: tails, head: tails, ...}

    l_tails = [tail for tail in tree.relation[node] if tail<node]
    r_tails = sorted([tail for tail in tree.relation[node] if tail>node],reverse=True)

    for lt in l_tails:
        ## x -> lt, x
        lt_governed = []
        dfs_for_span(tree, lt, lt_governed)
        tails = [lt,node]
        head_gov_set = set(lt_governed+tails)
        head_gov_span = [min(head_gov_set)-1,max(head_gov_set)]
        node_name = '_'.join(map(str,[node]+head_gov_span))
        new_edge = (node,tails,head_gov_span,node_name) #(1, [1, 5], [0, 9], '1_0_9')
        actions.append(new_edge)
        cfg_conversion(tree, lt, actions, visited)
    
    for rt in r_tails:
        ## x -> x, rt
        rt_governed = []
        dfs_for_span(tree, rt, rt_governed)
        tails = [node,rt]
        head_gov_set = set(rt_governed+tails)
        head_gov_span = [min(head_gov_set)-1,max(head_gov_set)]
        node_name = '_'.join(map(str,[node]+head_gov_span))
        new_edge = (node,tails,head_gov_span,node_name)
        actions.append(new_edge)
        cfg_conversion(tree, rt, actions, visited)        

def assign_cfg_id(hypo):

    ## find govening spans for each node in edges

    args:
        hypo: logp,edges,u,num_root
            hypo.edges: {(tail, head, label)}

    return:
        defaultdict(<class 'list'>, {1: [0, 9], 2: [1, 2], 3: [2, 3], 4: [2, 4], 5: [4, 9], 6: [5, 6], 7: [6, 7], 8: [7, 8], 9: [6, 9]})

    edges = sorted(list(hypo.edges), key=lambda x: x[1])

    span_ids = defaultdict(list)
    ## form a tree from edges
    tree = Tree(edges)
    ## find span_ids
    for node in range(1,len(edges)+1):
        governed=[]
        dfs_for_span(tree, node, governed)
        governed = sorted(governed)
        span_ids[node] = [governed[0]-1, governed[-1]]
    
    return span_ids

class Hyperedge:
    def __init__(self, head):
        self.tails = []
        self.probs = []
        self.labels = []
        self.head = head
        self.span_ids = [[],[]]

    def as_list(self):
        return [self.tails, self.head, self.labels, self.probs, self.span_ids]
    
    def sort_list(self):
        l = len(self.tails)
        self.tails.append(l**10)
        self.labels.append(l**10)
        
        i=0
        while i<l:
            j=1
            while j<l-i:
                if self.tails[i]>self.tails[i+j]:
                    self.tails[l]=self.tails[i]
                    self.tails[i]=self.tails[i+j]
                    self.tails[i+j]=self.tails[l]
                    self.tails[l]=l**10
                    self.labels[l]=self.labels[i]
                    self.labels[i]=self.labels[i+j]
                    self.labels[i+j]=self.labels[l]
                    self.labels[l]=l**10     
                    j+=1
                else:
                    j+=1
            i+=1
        self.tails.remove(l**10)
        self.labels.remove(l**10)

        return self.tails, self.labels

def form_hyperedges(hypo, parse_probs, rel_probs, span_ids):
    #hypo.edges
    #{(9, 6, 1), (3, 2, 1), (7, 9, 1), (6, 5, 1), (5, 4, 1), (8, 9, 1), (4, 2, 1)}

    ## create hyperedge representations from a set of edges
    # all elements in sorted order
    hyperedges = []
    edges = sorted(list(hypo.edges), key=lambda x: x[1])
    #print(edges)
    d_tails = defaultdict(list)
    d_labels = defaultdict(list)

    for edge in edges:
        d_tails[edge[1]].append(edge[0])
        d_labels[edge[1]].append(edge[2])

    for key in d_tails:
        he = Hyperedge(key)
        he.tails = d_tails[key]
        he.labels = d_labels[key]
        he.sort_list()
        he.probs = [str(parse_probs[tail,key]*rel_probs[tail,key,:][label]) for tail,label in zip(he.tails,he.labels)]

        for tail in he.tails:
            he.span_ids[0].append(span_ids[tail])
        he.span_ids[1].append(span_ids[key])

        hyperedges.append(he)

    return hyperedges

def assign_id(hypo):
    '''
    ## find govening spans for each node in edges

    args:
        hypo: logp,edges,u,num_root
            hypo.edges: {(tail, head, label)}

    return:
        defaultdict(<class 'list'>, {1: [0, 9], 2: [1, 2], 3: [2, 3], 4: [2, 4], 5: [4, 9], 6: [5, 6], 7: [6, 7], 8: [7, 8], 9: [6, 9]})
    '''
    edges = sorted(list(hypo.edges), key=lambda x: x[1])

    span_ids = defaultdict(list)
    ## form a tree from edges
    tree = Tree(edges)
    ## find span_ids
    for node in range(1,len(edges)+1):
        governed=[]
        dfs_for_span(tree, node, governed)
        governed = sorted(governed)
        span_ids[node] = [governed[0]-1, governed[-1]]
    
    return span_ids

def outside_rescore_function(md, hd, parse_probs, rescores, RESCORE, ALPHA):
    if RESCORE=='outside':
        if rescores[md][1] is not None and rescores[hd][1] is not None:
            ## md,hd are both verbs
            if int(ALPHA)==0:
                ## replace parse_probs with rescores
                score = rescores[md][1][hd-1]
            else:
                ## scores summed
                score = parse_probs[md,hd] + ALPHA*rescores[md][1][hd-1]
        else:
            ## either/both md,hd are verbs
            score = parse_probs[md,hd]
    else:
        score = parse_probs[md,hd]
    return score