import configparser
import os, sys
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from conll import pkl_loader, pruned_loader

sys.path.append('../')
import forest_rescorer_pruned_lb
from rescore_module.my_lib.rescore_all import RescoreModel

## paths
#fix hard-coding
model_name='NV-NV'#NV-NV
k=2
data='mytree'
pkl_dir='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl/unlabel2/'+str(data)+'/k'+str(k)
pruned_dir='/home/is/yuki-yama/work/d3/dep-forest-complex/inside-outside/out/unlabel2/'+str(data)+'/k'+str(k)
save_dir = '/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/model/o2/'+str(data)+'/'+model_name+'_k'+str(k)
os.makedirs(save_dir, exist_ok=True)

## load forests
_,_,_,all_sents,all_tags = pkl_loader(pkl_dir)
all_forests = pruned_loader(pruned_dir)

## load model
if model_name=='V-V':
    config_path = '/home/is/yuki-yama/work/d3/dep-forest-complex/rescore_module/o2_rescore.cfg'
elif model_name=='NV-NV':
    config_path = '/home/is/yuki-yama/work/d3/dep-forest-complex/rescore_module/o2_rescore_nv.cfg'
cfg_o2 = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
cfg_o2.read(config_path)
remodel_o2 = RescoreModel(cfg_o2)

def find_child(head_node_i, res, tmp):
    hi = id2Xspan[head_node_i][0]

    if node2edges[head_node_i]:
        for edge in node2edges[head_node_i]:
            child_node_i1, child_node_i2 = edge[:2]
            ci1,ci2 = id2Xspan[child_node_i1][0],id2Xspan[child_node_i2][0]

            if ci1==hi:
                tmp.append(ci2)
                find_child(child_node_i1, res, tmp)
            else:
                tmp.append(ci1)
                find_child(child_node_i2, res, tmp)
            res.append(list(sorted(tmp)))
            tmp = []

    return res

dl = []
## get all head-tails from forest
for i,(forest,sent,tags) in enumerate(tqdm(zip(all_forests,all_sents,all_tags))):
    pattern2nodes = defaultdict(list)
    hi2pat = defaultdict(list)
    rooted_sent = ['ROOT']+sent
    node2edges, id2Xspan, Xspan2id, topological_id = forest_rescorer_pruned_lb.load_pruned(forest)

    patterns = set()
    for head_node_i in list(reversed(topological_id)):
        hi = id2Xspan[head_node_i][0]
        children_l = find_child(head_node_i, [], [])
        
        ## linearize
        for children in children_l:
            node_list = [hi]+[children][0]
            bert_input_l = [rooted_sent[hi], '(',]+[rooted_sent[c] for c in children]+[')']
            pattern = ' '.join(bert_input_l)

            if pattern not in patterns:
                pattern2nodes[pattern] = node_list
                patterns.add(pattern)
    
    for k,v in pattern2nodes.items():
        hi = v[0]
        hi2pat[hi].append(k)

    dl.append((pattern2nodes,hi2pat))
    
## rescore
rescore_matrix = []
for i,(sent,tags) in tqdm(enumerate(zip(all_sents,all_tags))):
    pattern2nodes,hi2pat = dl[i]
    sent = ['ROOT']+sent
    tags = ['ROOT']+tags
    
    if model_name=='V-V':
        predicted = remodel_o2.head_prediction_V_o2(sent, tags, hi2pat)
    elif model_name=='NV-NV':
        predicted = remodel_o2.head_prediction_NV_o2(sent, tags, hi2pat)
    rescore_matrix.append(predicted)

    ## make pkl of dict (key:head-tails, value score)    
    with open(os.path.join(save_dir,'pred.pkl'), 'wb') as f:
        pkl.dump(rescore_matrix, f)
    with open(os.path.join(save_dir,'dicts.pkl'), 'wb') as g:
        pkl.dump(dl, g)    

sys.exit('debug')

