import numpy as np
from scipy.stats import chi2, chi2_contingency

from collections import defaultdict
from pprint import pprint

punct = set(['``', "''", ':', ',', '.', 'PU', 'PUNCT'])

def parse_gold(file_path):
    dependencies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                fields = line.strip().split('\t')
                head, dep, label = fields[6], fields[0], fields[7]
                if fields[3] not in punct:
                    dependencies.append((int(head), int(dep), label))
    return dependencies

def parse_conll(file_path):
    """
    Parse a CoNLL-U file and extract dependency relations
    """
    dependencies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                fields = line.strip().split('\t')
                head, dep, label = fields[6], fields[0], fields[7]
                if head == '_':
                    head, dep, label = fields[8], fields[0], fields[9]
                if fields[3] not in punct:
                    dependencies.append((int(head), int(dep), label))
    return dependencies

def load_snt(gold):
    snt = []
    with open(gold, 'r', encoding='utf-8') as f:
        tmp = []
        for i,line in enumerate(f):
            if line.strip() and not line.startswith('#'):
                fields = line.strip().split('\t')
                tmp.append(fields[1])
            else:
                if tmp:
                    snt.append(' '.join(tmp))
                    tmp = []
    return snt

def parse_file2(file2,snt):
    d = defaultdict()
    with open(file2, 'r', encoding='utf-8') as f:
        tmp = []
        tmp_l = []
        for i,line in enumerate(f):
            if line.strip() and not line.startswith('#'):
                fields = line.strip().split('\t')
                head, dep, label = int(fields[8]), int(fields[0]), fields[9]
                if fields[3] not in punct:
                    tmp_l.append((head, dep, label))
                tmp.append(fields[1])
            else:
                if tmp:
                    d[' '.join(tmp)] = tmp_l
                    tmp_l = []
                    tmp = []
    
    dependencies2 = []
    for s in snt:
        if d[s]:
            dependencies2.extend(d[s])
    
    return dependencies2

def compute(table):
    # Calculate McNemar's test statistic
    n12 = table[0, 1]  # Model 1 correct, Model 2 incorrect
    n21 = table[1, 0]  # Model 1 incorrect, Model 2 correct
    statistic = (n12 - n21)**2 / (n12 + n21)

    # Degrees of freedom for McNemar's test
    df = 1

    # Calculate the p-value using chi-square distribution
    p_value = 1 - chi2.cdf(statistic, df)

    #print("McNemar's test statistic:", statistic)
    #print("p-value:", p_value)

    return statistic, p_value

def mcnemar_test(gold, file1, file2):
    """
    Perform McNemar's test on two CoNLL-U files
    """
    gold_dep = parse_gold(gold)
    dependencies1 = parse_conll(file1)
    dependencies2 = parse_conll(file2)

    snt = load_snt(gold)
    dependencies2 = parse_file2(file2,snt)
    #pprint(dependencies2)

    print(len(gold_dep),len(dependencies1),len(dependencies2))
    
    # Create contingency table
    table_uas = np.zeros((2, 2))
    for gold, dep1, dep2 in zip(gold_dep, dependencies1, dependencies2):
        gold, dep1, dep2 = gold[:2], dep1[:2], dep2[:2]
        if dep1 == gold and dep2 != gold:
            table_uas[0, 1] += 1  ## Model 1 correct, Model 2 incorrect
        elif dep1 != gold and dep2 == gold:
            table_uas[1, 0] += 1  ## Model 1 incorrect, Model 2 correct
        elif dep1 != gold and dep2 != gold:
            table_uas[1, 1] += 1  ## Both models incorrect
        else:
            table_uas[0, 0] += 1  ## Both models correct

    table_las = np.zeros((2, 2))
    for gold, dep1, dep2 in zip(gold_dep, dependencies1, dependencies2):
        if dep1 == gold and dep2 != gold:
            table_las[0, 1] += 1  ## Model 1 correct, Model 2 incorrect
        elif dep1 != gold and dep2 == gold:
            table_las[1, 0] += 1  ## Model 1 incorrect, Model 2 correct
        elif dep1 != gold and dep2 != gold:
            table_las[1, 1] += 1  ## Both models incorrect
        else:
            table_las[0, 0] += 1  ## Both models correct

    # Perform McNemar's test
    '''
    chi2_uas, p_value_uas = chi2_contingency(table_uas, correction=False)[:2]
    print(table_uas)
    print(chi2_uas, p_value_uas)
    chi2_las, p_value_las = chi2_contingency(table_las, correction=False)[:2]
    print(table_las)
    print(chi2_las, p_value_las)
    '''
    
    stat_uas, p_value_uas = compute(table_uas)
    stat_las, p_value_las = compute(table_las)
    
    return stat_uas, p_value_uas, stat_las, p_value_las, table_uas, table_las

# Example usage

'''
gold='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/data/wsj_sd_cophead/test.conllu'
file1='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/saves/ptb_cophead/test.conllu_1best.txt'
file2='/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/pruned/unlabel2/V-Any_NV-NV/test/lb_mod/k16/rescore_3-01-01.conllu'
'''
gold='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/data/wsj_sd_cophead/mytree_upos.conllu'
file1='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/saves/ptb_cophead/mytree_upos.conllu_1best.txt'
file2='/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/pruned/unlabel2/V-Any/mytree/lb_mod/k16/rescore_3-01-01.conllu'
file2='/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/vanilla/mytree/k16/vanilla_3.conllu'

'''
gold='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/data/ctb5.1/test.conllx'
file1='/home/is/yuki-yama/work/d3/dep-forest-complex/biaffine_forest/saves/ctb5.1/test.conllx_1best.txt'
file2='/home/is/yuki-yama/work/d3/dep-forest-complex/outputs/vanilla/ctb5.1/k16/vanilla_3.conllu'
'''

chi2_uas, p_value_uas, chi2_las, p_value_las, table_uas, table_las = mcnemar_test(gold, file1, file2)
'''
print('UAS')
print("Chi-squared statistic:", chi2_uas)
print("p-value:", p_value_uas)
print('LAS')
print("Chi-squared statistic:", chi2_las)
print("p-value:", p_value_las)
'''
print('UAS')
print(table_uas)
print("statistic:", chi2_uas)
print("p-value:", p_value_uas)
print('LAS')
print(table_las)
print("statistic:", chi2_las)
print("p-value:", p_value_las)

