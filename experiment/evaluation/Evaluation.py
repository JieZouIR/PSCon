import codecs
import json
from ndcg import calculate_ndcg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluation.Rouge import *
from collections import defaultdict
import numpy as np
from scipy import stats
import os
min_num = 1e-8
from sklearn.metrics import ndcg_score, average_precision_score

def evaluate_T1(rs_t1, gt_t1):
    # rs_t1 contains predicted results
    # gt_t1 contains ground truth results
    # macro
    count_correct = {}
    count_rs = {}
    count_gt = {}
    for id in gt_t1:
        gt='-'.join(gt_t1[id]['intent'])
        if gt not in count_gt:
            count_gt[gt]=1
        else:
            count_gt[gt] += 1
        if id in rs_t1:
            rs = '-'.join(rs_t1[id]['intent'])
            if rs not in count_rs:
                count_rs[rs]=1
            else:
                count_rs[rs] += 1
            if rs==gt:
                if rs not in count_correct:
                    count_correct[rs]=1
                else:
                    count_correct[rs] += 1
    precision_dict={}
    recall_dict={}
    f1_dict={}
    for intent in count_gt:
        correct=count_correct.get(intent, 0)
        p=0
        r=0
        if intent in count_rs:
            p=float(correct) / count_rs[intent]
            precision_dict[intent] = p
        if intent in count_gt:
            r=float(correct) / count_gt[intent]
            recall_dict[intent] = r
        f1_dict[intent] = 2.*(p * r / (p + r + min_num))
    precision=0
    recall=0
    f1=0
    for intent in precision_dict:
        precision+=precision_dict[intent]
    precision=precision/float(len(precision_dict))
    for intent in recall_dict:
        recall+=recall_dict[intent]
    recall=recall/float(len(recall_dict))
    for intent in f1_dict:
        f1+=f1_dict[intent]
    f1=f1/float(len(f1_dict))
    return {'t1_precision': precision_dict, 't1_recall': recall_dict, 't1_f1': f1_dict, 't1_avg_precision': precision, 't1_avg_recall':recall, 't1_avg_f1':f1}


def evaluate_T2(rs_t2, gt_t2, tokenizer):
    actions = ['Clarify', 'Recommend']

    bleu_dict = {}
    rouge_dict={}
    count_dict={}
    for id in gt_t2:
        if id not in rs_t2:
            continue
        gt_action=gt_t2[id]['action']
        if gt_action[0] in actions:
            gt = [tokenizer(s) for s in gt_t2[id]['state']]
            if not gt:
                continue
            b=0
            r=0
            for s in rs_t2[id]['state']:
                b+=sentence_bleu(gt, s, weights=(1.,), smoothing_function=SmoothingFunction().method2)
                r+=sentence_rouge(gt, s)

            gt_action='-'.join(gt_action)
            if len(rs_t2[id]['state'])>0:
                b=b/len(rs_t2[id]['state'])
                r=r / len(rs_t2[id]['state'])
                if gt_action not in bleu_dict:
                    bleu_dict[gt_action] = b
                else:
                    bleu_dict[gt_action] += b
                if gt_action not in rouge_dict:
                    rouge_dict[gt_action] = r
                else:
                    rouge_dict[gt_action] += r
                if gt_action not in count_dict:
                    count_dict[gt_action] = 1
                else:
                    count_dict[gt_action] += 1

    for gt_action in bleu_dict:
        bleu_dict[gt_action] /= count_dict[gt_action]
        rouge_dict[gt_action] /= count_dict[gt_action]

    bleu=0
    rouge=0
    for action in bleu_dict:
        bleu += bleu_dict[action]
    bleu = bleu / float(len(bleu_dict) + 1)
    for action in rouge_dict:
        rouge += rouge_dict[action]
    rouge = rouge / float(len(rouge_dict) + 1)
    return {'t2_avg_bleu_1': bleu, 't2_avg_rouge_l': rouge, 't2_bleu_1': bleu_dict, 't2_rouge_l': rouge_dict}

def evaluate_T3(rs_t3, gt_t3):
    count_correct={}
    count_rs={}
    count_gt={}
    for id in gt_t3:
        gt='-'.join(gt_t3[id]['action'])
        if gt not in count_gt:
            count_gt[gt]=1
        else:
            count_gt[gt] += 1
        if id in rs_t3:
            rs = '-'.join(rs_t3[id]['action'])
            if rs not in count_rs:
                count_rs[rs] = 1
            else:
                count_rs[rs] += 1
            if rs == gt:
                if rs not in count_correct:
                    count_correct[rs] = 1
                else:
                    count_correct[rs] += 1
    precision_dict={}
    recall_dict={}
    f1_dict={}
    for action in count_gt:
        correct=count_correct.get(action, 0)
        p=0
        r=0
        if action in count_rs:
            p=float(correct) / count_rs[action]
            precision_dict[action] = p
        if action in count_gt:
            r=float(correct) / count_gt[action]
            recall_dict[action] = r
        f1_dict[action] = 2. * (p * r / (p + r + min_num))

    precision = 0
    recall = 0
    f1 = 0
    for action in precision_dict:
        precision += precision_dict[action]
    precision = precision / float(len(precision_dict) + 1)
    for action in recall_dict:
        recall += recall_dict[action]
    recall = recall / float(len(recall_dict) + 1)
    for action in f1_dict:
        f1 += f1_dict[action]
    f1 = f1 / float(len(f1_dict) + 1)
    return {'t3_precision': precision_dict, 't3_recall': recall_dict, 't3_f1': f1_dict, 't3_avg_precision': precision, 't3_avg_recall':recall, 't3_avg_f1':f1}

def evaluate_T4(rs_t4, gt_t4):
    actions=['Clarify', 'Recommend']
    #actions=['Clarify']

    pred_list = []
    label_list = []

    gt_len = 0
    ndcg10 = 0
    map = 0
    ndcgs = []
    for id in gt_t4:
        gt_action=gt_t4[id]['action']
        if id not in rs_t4:
            continue
        if gt_action[0] in actions and len(gt_t4[id]['selected_query'])>0:            
            rs_query_ranking = rs_t4[id]['query_ranking']
            gt_selected_query = gt_t4[id]['selected_query']
            label = []
            pred = []
            # 创建标签列表，标签为1表示在gt_selected_passage内，0表示不在
            #print(f"rs_query_ranking:\n{rs_query_ranking}\n")
            #print(f"gt_query_passage:\n{gt_selected_query}\n")
            # for each in rs_passage_ranking:
            #     for key, value in each.items():
            #         print(str(key), value)
            #         if str(key) in gt_selected_passage:
            #             label.append(1)
            #         else:
            #             label.append(0)
            #         pred.append(value)
            for each in rs_query_ranking:
                key, value = each[0] , each[1]
                #print(str(key), value)
                if str(key) in gt_selected_query:
                    label.append(1)
                else:
                    label.append(0)
                pred.append(value)

            # label = [1 if passage in gt_selected_passage else 0 for passage in rs_passage_ranking]
            # pred = [list(entry.values())[0] for entry in rs_passage_ranking]
            #print(f"label:{label}")
            #print(f"pred:{pred}")
            if 1 in label:
                gt_len = gt_len + 1
                label_list.append(label)
                pred_list.append(pred)
                label = [label]
                pred = [pred]
                ndcgi = ndcg_score(label, pred)
                ndcgs.append(ndcgi)
                ndcg10 = ndcg10 + ndcgi
                # map = map + average_precision_score(label, pred)
    
    print(label_list)
    print(pred_list)
    # print(ndcgs)
    ndcg_list = ndcg_score(label_list, pred_list)
    map_score = average_precision_score(label_list, pred_list)
    ndcg10 = ndcg10 / gt_len
    # map = map / gt_len
    mrr_list = []
    for labels, preds in zip(label_list, pred_list):
        ranks = []
        for i, label in enumerate(labels):
            if label == 1:
                # rank = np.reciprocal(preds[i]) if preds[i] != 0 else 0  # Calculate reciprocal rank for each relevant label
                ranks.append(1 / (i+1))
        
        if ranks:
            mrr_list.append(np.mean(ranks))  # Calculate mean of reciprocal ranks for the sample
        else:
            mrr_list.append(0.0)
    mrr = np.mean(mrr_list)

    ndcg_new = calculate_ndcg(label_list, pred_list, 10)
    
    return {'t4_ndcg': ndcg_list, 'map': map_score,  't4_mrr': mrr}
    # return {'t4_ndcg': ndcg10, 'ndcg_list': ndcg_list ,  't4_mrr': mrr}

def evaluate_T5(rs_t5, gt_t5):
    actions=['Recommend']

    label_list = []
    pred_list = []

    for id in gt_t5:
        gt_action=gt_t5[id]['action']
        if id not in rs_t5:
            continue
        if gt_action[0] in actions and len(gt_t5[id]['selected_passage'])>0:
            rs_passage_ranking = rs_t5[id]['passage_ranking']
            gt_selected_passage = gt_t5[id]['selected_passage']
            label = []
            pred = []
            # 创建标签列表，标签为1表示在gt_selected_passage内，0表示不在
            #print(f"rs_passage_ranking:\n{rs_passage_ranking}\n")
            #print(f"gt_selected_passage:\n{gt_selected_passage}\n")
            # for each in rs_passage_ranking:
            #     for key, value in each.items():
            #         print(str(key), value)
            #         if str(key) in gt_selected_passage:
            #             label.append(1)
            #         else:
            #             label.append(0)
            #         pred.append(value)
            for each in rs_passage_ranking:
                key, value =each[0],each[1]
                #print(str(key), value)
                if str(key) in gt_selected_passage:
                    label.append(1)
                else:
                    label.append(0)
                pred.append(value)
            # label = [1 if passage in gt_selected_passage else 0 for passage in rs_passage_ranking]
            # pred = [list(entry.values())[0] for entry in rs_passage_ranking]
            #print(f"label:{label}")
            #print(f"pred:{pred}")
            label_list.append(label)
            pred_list.append(pred)

    #ndcg1 = ndcg_score(label_list, pred_list,1)
    #ndcg2 = ndcg_score(label_list, pred_list,2)
    #ndcg3 = ndcg_score(label_list, pred_list,3)
    #ndcg4 = ndcg_score(label_list, pred_list,4)
    #ndcg5 = ndcg_score(label_list, pred_list,5)
    ndcg10 = ndcg_score(label_list, pred_list)
    
    map_score = average_precision_score(label_list, pred_list)
    
    mrr_list = []
    # for label, pred in zip(label_list, pred_list):
    #     rank = 0
    #     for i in range(len(label)):
    #         if label[i] == 1:
    #             rank = i + 1
    #             break
    #     mrr_list.append(1 / rank )
    for labels, preds in zip(label_list, pred_list):
        ranks = []
        for i, label in enumerate(labels):
            if label == 1:
                # rank = np.reciprocal(preds[i]) if preds[i] != 0 else 0  # Calculate reciprocal rank for each relevant label
                ranks.append(1 / (i+1))
        
        if ranks:
            mrr_list.append(np.mean(ranks))  # Calculate mean of reciprocal ranks for the sample
        else:
            mrr_list.append(0.0)
    mrr = np.mean(mrr_list)

    ndcg_new = calculate_ndcg(label_list, pred_list, 10)

    return {'t5_ndcg': ndcg10,  't5_map': map_score, 't5_mrr': mrr}

def evaluate_T6(rs_t6, gt_t6, tokenizer):
    actions = ['Clarify', 'Recommend']

    bleu_dict={}
    rouge_dict={}
    count_dict={}
    for id in gt_t6:
        gt_action=gt_t6[id]['action']
        if gt_action[0] in actions:
            if id not in rs_t6:
                continue
            rs=rs_t6[id]['response']
            gt=tokenizer(gt_t6[id]['response'])
            gt_action='-'.join(gt_action)

            b= sentence_bleu([gt], rs, weights=(1.,), smoothing_function=SmoothingFunction().method2)
            r= sentence_rouge([gt], rs)
            if gt_action not in bleu_dict:
                bleu_dict[gt_action] = b
            else:
                bleu_dict[gt_action] += b
            if gt_action not in rouge_dict:
                rouge_dict[gt_action] = r
            else:
                rouge_dict[gt_action] += r
            if gt_action not in count_dict:
                count_dict[gt_action] = 1
            else:
                count_dict[gt_action] += 1
    for gt_action in bleu_dict:
        bleu_dict[gt_action] /= count_dict[gt_action]
        rouge_dict[gt_action] /= count_dict[gt_action]
    bleu=0
    rouge=0
    for action in bleu_dict:
        bleu += bleu_dict[action]
    bleu = bleu / float(len(bleu_dict) + 1)
    for action in rouge_dict:
        rouge += rouge_dict[action]
    rouge = rouge / float(len(rouge_dict) + 1)

    return {'t6_avg_bleu_1': bleu, 't6_avg_rouge_l': rouge, 't6_bleu_1': bleu_dict, 't6_rouge_l': rouge_dict}

def evaluate(rs_files, gt_files, tokenizer):
    rs={}
    for file in rs_files:
        file_name = os.path.basename(file)
        #print(file_name)
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = conv[-i-1]['intent']
                        break
                selected_query = conv[-1]['selected_query']
                selected_passage = conv[-1]['selected_passage']
                rs[conv[-1]['msg_id']]={'msg_id': conv[-1]['msg_id'], 'intent':intent, 'state':conv[-1]['state'], 'action':conv[-1]['action'], 'query_ranking':selected_query, 'passage_ranking':selected_passage, 'response':conv[-1]['response']}

    gt={}
    for file in gt_files:
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = conv[-i-1]['intent']
                        break
                selected_query = conv[-1]['selected_query']
                selected_passage = conv[-1]['selected_passage']
                gt[conv[-1]['msg_id']]={'msg_id': conv[-1]['msg_id'], 'intent':intent, 'state':conv[-1]['state'], 'action':conv[-1]['action'], 'selected_query':selected_query, 'selected_passage':selected_passage, 'response':conv[-1]['response']}

    t1=evaluate_T1(rs, gt)
    t2=evaluate_T2(rs, gt, tokenizer)
    t3=evaluate_T3(rs, gt)
    t4=evaluate_T4(rs, gt)
    t5=evaluate_T5(rs, gt)
    t6=evaluate_T6(rs, gt, tokenizer)

    current_directory = os.getcwd()
    output_file_path = os.path.join(current_directory, "rsgt.txt")
    with open(output_file_path, "w") as file:
        file.write(f"rs:{rs}\n")
        file.write(f"gt:{gt}\n")

    return {**t1, **t2, **t3, **t4, **t5, **t6}
