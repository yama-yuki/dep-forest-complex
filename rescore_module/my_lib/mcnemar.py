import argparse
import csv
import sys
from tqdm import tqdm
from statsmodels.stats.contingency_tables import mcnemar

parser = argparse.ArgumentParser()
parser.add_argument('--pred_1_path')
parser.add_argument('--pred_2_path')
args = parser.parse_args()

def main():
    pred_1 = pred_2 = []
    a = b = c = d = 0

    with open(args.pred_1_path, mode='r',encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for row in reader:
            pred_1 = row
    with open(args.pred_2_path, mode='r',encoding='utf-8') as f2:
        reader = csv.reader(f2)
        for row in reader:
            pred_2 = row
    
    if len(pred_1) == len(pred_2):
        for i in tqdm(range(len(pred_1))):
            if (pred_1[i],pred_2[i]) == ('1','1'):
                a+=1
            elif (pred_1[i],pred_2[i]) == ('1','0'):
                b+=1
            elif (pred_1[i],pred_2[i]) == ('0','1'):
                c+=1
            else:
                d+=1

    else:
        sys.exit('inconsistant length')

    data = [[a,b],[c,d]]
    print(data)
    print(mcnemar(data, exact=False))


if __name__ == '__main__':
    main()