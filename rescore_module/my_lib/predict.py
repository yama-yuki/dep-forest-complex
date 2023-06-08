import argparse
import csv
import json
import logging
import os
import sys
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig#BertForQuestionAnswering
from tqdm import tqdm
from evalq import evaluate, evaluate_analysis

logger = logging.getLogger('EVAL_LOG')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def main():
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--model_dir')
    parser.add_argument('--pretrained', default='bert-base-uncased')
    parser.add_argument('--input_path', help='.json')
    parser.add_argument('--output_path', help='.json')
    parser.add_argument('--test_path', help='.json')
    parser.add_argument('--analysis_path')
    parser.add_argument('--mcnemar', action='store_true')
    args = parser.parse_args()

    mode = args.mode
    model_directory = args.model_dir
    pretrained_model = args.pretrained

    ## load model & tokenizer
    config = AutoConfig.from_pretrained(os.path.join(model_directory, "config.json"))
    tokenizer_config = AutoConfig.from_pretrained(os.path.join(model_directory, "tokenizer_config.json"))

    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model, config=config)
    model.load_state_dict(torch.load(os.path.join(model_directory, "pytorch_model.bin"), map_location=torch.device('cuda')))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, config=tokenizer_config)

    def predict(question, context):

        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        output = model(**inputs)
        answer_start = torch.argmax(output.start_logits)  
        answer_end = torch.argmax(output.end_logits) + 1 
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer

    ## mode specification
    logger.info('MODE: '+mode)
    if mode == 'pred':
        ## predict & write out
        in_path = args.input_path
        out_path = args.output_path

        with open(in_path,mode='r',encoding='utf-8') as f:
            with open(out_path,mode='w',encoding='utf-8') as o:
                reader = csv.reader(f, delimiter='\t')
                writer = csv.writer(o, delimiter='\t')
                
                for i,row in enumerate(reader):
                    logger.info(row)
                    questions = row[0:-1]
                    context = row[-1]
                    logger.info('#'+str(i))
                    logger.info('Q: '+str(*questions))
                    answers = [predict(question, context) for question in questions]
                    logger.info('A: '+str(*answers))
                    writer.writerow(answers)
    
    elif mode == 'test':
        ## evaluate on test set
        test_path = args.test_path
        pred_path = 'pred.json'

        logger.info('LOADING: '+test_path)
        with open(test_path, mode='r', encoding='utf-8') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']

        def make_pred_json(dataset):
            predictions = {}

            for entry in tqdm(dataset[0]['paragraphs']):
                ## split data             
                context = entry['context']
                q_list = [qa['question'] for qa in entry['qas']] #qa['answers'][0]['text']
                id_list = [qa['id'] for qa in entry['qas']]

                ## predict
                preds = [predict(q, context) for q in q_list]

                ## adding to new_entry
                for pred,idx in zip(preds,id_list):
                    predictions[idx] = pred

                ##{'context': "Do n't cross the bridge until you come to it is an English language proverb that is rich in metaphor .", 'qas': [{'answers': [{'answer_start': 68, 'text': 'proverb'}], 'question': 'come', 'id': '720473'}, {'answers': [{'answer_start': 7, 'text': 'cross'}], 'question': 'proverb', 'id': '720474'}]}

            return predictions

        predictions = make_pred_json(dataset)
        #sys.exit()

        with open(pred_path, mode='w', encoding='utf-8') as g:
            json.dump(predictions, g)

        if args.analysis_path:
            t_path = args.analysis_path+'_t.tsv'
            f_path = args.analysis_path+'_f.tsv'
            result, t_list, f_list = evaluate_analysis(dataset, predictions)

            with open(t_path, mode='w', encoding='utf-8') as t:
                writer = csv.writer(t, delimiter='\t')
                writer.writerows(t_list)
            with open(f_path, mode='w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(f_list)

        elif args.mcnemar:
            pred_01_list = evaluate_analysis(dataset, predictions, args.mcnemar)
            model_name = os.path.split(model_directory)[-1]
            _, test_data_name = os.path.split(test_path)
            pred_dir = os.path.join('../pred',model_name)
            save_path = os.path.join(pred_dir,test_data_name.split('.')[0]+'.csv')

            os.makedirs(pred_dir, exist_ok=True)
            
            with open(save_path, mode='w', encoding='utf-8') as o:
                writer = csv.writer(o)
                writer.writerow(pred_01_list)    
        
        else:
            del pred_path
            print(json.dumps(evaluate(dataset, predictions)))




    else:
        sys.exit('SPECIFY MODE')

## {"data": [{"title": "None", "paragraphs": 
## [{ "context": <context>, "qas": [
## {"answers": [{"answer_start": <id>, "text": <ans>}], "question": <question>, "id": <id>},

if __name__ == '__main__':
    main()