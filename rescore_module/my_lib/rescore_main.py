'''
## word2word rescoring model
input:
    sent:(list)
        ['Several', 'traders', 'could', 'be', 'seen', 'shaking', 'their', 'heads', 'when', 'the', 'news', 'flashed', '.']
    tags:(list)
        ['ROOT', 'JJ', 'NNS', 'MD', 'VB', 'VBN', 'VBG', 'PRP$', 'NNS', 'WRB', 'DT', 'NN', 'VBD', '.', 'PAD', 'PAD', 'PAD']
    cur_node:(str)
        'seen'

output:
    rescores:(ndarray)
        {'1': (['several', 'traders', 'could', 'be', 'seen', 'shaking', 'their', 'heads', 'when', 'the', 'news', 'flashed', '.'], 
        array([7.3567807e-04, 4.8336340e-04, 4.0837805e-04, 5.3011143e-04,
        2.7120281e-02, 3.5011023e-01, 3.8721948e-04, 6.5450097e-04,
        8.4272813e-04, 5.9153832e-04, 6.6240935e-04, 6.1724335e-01,
        2.3023642e-04], dtype=float32))}
'''

import json
import os
import sys
from tqdm import tqdm
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../rescore/rescore'))

import torch
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig#BertForQuestionAnswering
import numpy as np

import configparser

import logging
logger = logging.getLogger('EVAL_LOG')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR,)

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
hide_pb = True

class RescoreModel:
    def __init__(self, cfg):
        super().__init__()

        self.cfg=cfg

        ## read config
        self.config = AutoConfig.from_pretrained(os.path.join(self.cfg['OS']['model_dir'], "config.json"))
        self.tokenizer_config = AutoConfig.from_pretrained(os.path.join(self.cfg['OS']['model_dir'], "tokenizer_config.json"))

        ## load model & tokenizer
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.cfg['Model']['pretrained'], config=self.config)
        self.model.load_state_dict(torch.load(os.path.join(self.cfg['OS']['model_dir'], "pytorch_model.bin"), map_location=torch.device('cuda:0')))
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['Model']['pretrained'], config=self.tokenizer_config)

        self.model.eval()

    def get_embedding(self, snt1, snt2):
        tokenized_snt = self.tokenizer(snt1, snt2)
        indexed_tok = self.tokenizer.convert_tokens_to_ids(tokenized_snt)

        print(tokenized_snt)
        print(indexed_tok)
        tensor_tok = torch.tensor([indexed_tok])
        with torch.no_grad():
            all_encoder_layers = self.model(tensor_tok)
        #print(all_encoder_layers)
        #embedding = all_encoder_layers[1][-2].numpy()[0]
        #result = np.mean(embedding,axis=0)

        return tensor_tok
    
    def head_prediction(self, sent, tags):
        sent_predictions = []
        '''
        [None, None, None, array([1.9267636e-05, 1.4064037e-05, 1.7708957e-05, 2.1448202e-05,...]
        '''

        cnt=0
        for cur_node,tag in zip(sent,tags):

            if tag[0] == 'V':
                ##create input to rescore model (.json)
                self._to_squad(sent, cur_node, self.cfg['OS']['input_path'])
                ## predict & return score
                predictions = self._predict_head()
                '''predictions
                {'4': (['that', 'debt', 'would', 'be', 'paid'], array([2.8575512e-08, 2.8993115e-08, 2.7188630e-08, 2.8511652e-08, 9.9998128e-01], dtype=float32))}
                '''
                print('predictions V'+str(cnt))
                print('predictions:'+str(predictions.size))
                print(predictions)
                sent_predictions.append(predictions)
                cnt+=1
            else:
                sent_predictions.append(None)

        return sent_predictions

    def head_prediction_lintree(self, sent, md, hd, lintree):
        ##tail_top,head_top = sent[md-1],sent[hd-1]
        self._to_squad(sent, lintree, self.cfg['OS']['input_path'])
        rescore_matrix = self._predict_head()
        bert_score = rescore_matrix[hd-1]

        return bert_score

    def _to_squad(self, sent, cur_node, input_path):
        new_data = {"data": [{"title": "None", "paragraphs":[]}]}
        entry = {"context":"","qas":[]}

        entry["context"] = ' '.join(sent)
        qas = []
        d = {"answers": [{"answer_start": "", "text": ""}], "question": "", "id": ""}
        d["question"] = cur_node
        d["id"] = str(1)
        qas.append(d)

        entry["qas"] = qas
        new_data["data"][0]["paragraphs"].append(entry)
        
        with open(input_path, 'w') as o:
            json.dump(new_data, o)

    def _predict_head(self):
        mode = self.cfg['Mode']['mode']
        ## mode specification
        logger.info('MODE: '+mode)
        input_path = self.cfg['OS']['input_path']
        pred_path = self.cfg['OS']['output_path'] #'pred.json'
        logger.info('LOADING: '+input_path)
        with open(input_path, mode='r', encoding='utf-8') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        if mode == 'pred':
            predictions = self._make_pred(dataset, mode)
            return predictions
        else:
            sys.exit('SPECIFY MODE')

    def _make_pred(self, dataset, mode):
        predictions = []
        for entry in tqdm(dataset[0]['paragraphs'], disable=hide_pb):
            ## split data             
            context = entry['context']
            q_list = [qa['question'] for qa in entry['qas']] #qa['answers'][0]['text']
            id_list = [qa['id'] for qa in entry['qas']]
            if mode == 'pred':
                ## predict
                preds = [self._predict(q, context, 'score') for q in q_list]
                predictions = preds[0][1]
                return predictions
            else:
                sys.exit('SPECIFY MODE')
        return predictions
    
    def _predict(self, question, context, mode):
        ##
        context_start_char=[]
        char_cnt=0
        for tok in context.split(' '):
            context_start_char.append(char_cnt)
            char_cnt+=(len(tok)+1)
        inputs = self.tokenizer.encode_plus(question, context, return_offsets_mapping=True, add_special_tokens=True,  return_tensors="pt")

        ## offsets
        offsets = torch.flatten(inputs['offset_mapping'], end_dim=1).tolist()
        start_offset, end_offset = self._count_offset(offsets)
        del inputs['offset_mapping']
        input_ids = inputs["input_ids"].tolist()[0]
        output = self.model(**inputs)

        if mode == 'ans':
            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits) + 1 
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            return answer
        
        elif mode == 'score':
            ## search bos_i: start id of sent2
            for i,idx in enumerate(inputs['token_type_ids'][0]):
                if idx == 1:
                    bos_i = i
                    break
            try:
                bos_i
            except NameError:
                sys.exit('bos_i Not Found')

            ## slice output
            tok, score = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids][bos_i:-1], output.start_logits[0][bos_i:-1]
            ans_i = torch.argmax(score)

            ##
            pooled_offsets = []
            pooling_marker = [1]
            prev = offsets[start_offset:end_offset+1][0]
            for i,curr in enumerate(offsets[start_offset:end_offset+1]):
                if i==0:
                    continue
                if curr[0] == prev[1]: ##前offset末が今offset頭と一致
                    prev = [prev[0], curr[1]]
                    pooling_marker.append(0)
                else: ##前offsetと今offsetが無関係
                    pooled_offsets.append(prev)
                    prev = curr
                    pooling_marker.append(1)
            pooled_offsets.append(prev)
            tok, logit_score = self._offset_pooling(tok, score, pooling_marker)

            ## softmax
            softmax_score = self._softmax(logit_score.to('cpu').detach().numpy().copy())
            
            return tok, softmax_score
        
        else:
            sys.exit('SPECIFY MODE')

    def _offset_pooling(self, tok, score, pooling_marker):
        new_tok = []
        new_score = torch.Tensor([])
        tmp_score = torch.Tensor([])
        tmp_tok = ''
        pooling_marker.append(-1)

        for i in range(len(tok)):
            if pooling_marker[i]*pooling_marker[i+1] in {-1,1}: ## 1,1/-1
                new_tok.append(tok[i])
                new_score = torch.cat((new_score, torch.unsqueeze(score[i],0)))
                continue

            if tok[i][:2] == '##':
                tmp_tok+=tok[i][2:]
            else:
                tmp_tok+=tok[i]

            if pooling_marker[i+1] == 0: ## *,0
                tmp_score = torch.cat((tmp_score, torch.unsqueeze(score[i],0)))
            else: #pooling_marker[i] == 0 and abs(pooling_marker[i+1]) == 1:
                pooled = torch.Tensor([torch.mean(tmp_score)])
                new_score = torch.cat((new_score, pooled))
                new_tok.append(tmp_tok)
                tmp_score = torch.Tensor([])  
                tmp_tok = ''

        return [new_tok, new_score]

    def _count_offset(self, offsets):
        cnt=0
        for i,offset in enumerate(offsets):
            if offset == [0,0]:
                if cnt == 1:
                    start_offset,start_char = i+1,offsets[i+1]
                elif cnt==2:
                    end_offset,end_char = i-1,offsets[i-1]
                cnt+=1
        try:
            all([bool(start_offset in locals()),bool(end_offset in locals())])
        except NameError:
            sys.exit('start_offset or end_offset Not Found')
        return start_offset, end_offset

    def _softmax(self, a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

if __name__ == '__main__':
    sent = ['Several', 'traders', 'could', 'be', 'seen', 'shaking', 'their', 'heads', 'when', 'the', 'news', 'flashed', '.']
    tags = ['JJ', 'NNS', 'MD', 'VB', 'VBN', 'VBG', 'PRP$', 'NNS', 'WRB', 'DT', 'NN', 'VBD', '.']
    #sent = ['``', 'that', "'s", 'good', 'news', ',', 'because', 'we', 'all', '<UNK>', 'in', 'this', 'water', '.', "''"]
    #tags = ['JJ', 'NNS', 'MD', 'VB', 'VBN', 'VBG', 'PRP$', 'NNS', 'WRB', 'DT', 'NN', 'VBD', 'AA', 'AA', '.']
    
    sent = ['Several', 'traders', 'could', 'be', 'seen', 'shaking', 'their', 'heads', 'when']
    tags = ['JJ', 'NNS', 'MD', 'VB', 'VBN', 'VBG', 'PRP$', 'NNS', 'WRB']
    verb_nodes = [bool(tag[0] == 'V') for tag in tags]

    cfg_path='../rescore.cfg'
    cfg = configparser.RawConfigParser()
    if not os.path.exists(cfg_path):
      logger.error('rescore.cfg not found')
      sys.exit(-1)
    cfg.read(cfg_path)

    remodel = RescoreModel(cfg)
    print(remodel.head_prediction(sent, tags))

    '''
    snt1 = 'flashed'
    snt2 = ' '.join(sent)
    print(remodel.get_embedding(snt1, snt2))
    '''

