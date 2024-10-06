import pickle as pkl
from tqdm import tqdm

if __name__ == '__main__':
    pkl_path = '/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/pkl/deprel.pkl'
    with open(pkl_path, 'rb') as p:
        deprel = pkl.load(p)
    print(deprel)

    model_name='V-Any_NV-NV'#V-Any_NV-NV
    data_type='test'
    a='03'
    b='01'

    conll_path = '/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/pruned/unlabel2/'+model_name+'/'+data_type+'/lb/k16/rescore_3-'+a+'-'+b+'.conllu'
    save_path = conll_path+'.lb'

    with open(save_path, mode='w', encoding='utf-8') as o:
        with open(conll_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                #print(line)
                sp = line.rstrip('\n').split('\t')
                if len(sp) == 10:
                    dep_id = sp[7]
                    label = deprel[int(dep_id)]
                    sp[9] = label
                    new_line = '\t'.join([str(i) for i in sp])+'\n'
                    o.write(new_line)
                else:
                    o.write(line)
    
