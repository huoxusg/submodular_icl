import json
import ast
from collections import defaultdict


def get_ner_stats(lines):
    stats_dict = {'pred': 0, 'label': 0, 'correct': 0}
    for line in lines:
        if isinstance(line, str):
            res = json.loads(line.strip())
        else:
            res = line
        
        if res['label']== 'None':
            entities = set()
        else:
            entities = set(ast.literal_eval(res['label']))
        
        stats_dict['label'] += len(entities)
        
        try:
            prediction = res['prediction'].split('\n\n### Response: \n')[1].split('\n')[0].strip()
            # if prediction.startswith('[') and not prediction.endswith(']'):
            #     prediction += ']'
            if prediction == 'None':
                continue
            # prediction = ast.literal_eval(prediction)
            # pred_entities = set(prediction)
            
            prediction = prediction.replace("['", "").replace("']", "")
            prediction = prediction.replace("[", "").replace("]", "")
            prediction = prediction.replace("',", ",").replace(" '", " ")
            pred_entities = set(prediction.split(', '))
            # print(pred_entities)
            
            stats_dict['pred'] += len(pred_entities)
            stats_dict['correct'] += len(entities.intersection(pred_entities))
            # if ii > 20:
            #     break
            
        except:
            # print('One item failed')
            # print(label, len(label), len(ast.literal_eval(res['input'])))
            # print(prediction)
            # print('###########')
            continue
    return stats_dict


def calculate_ner_metrics(stats_dict):
    prec, recall, f1 = 0, 0, 0
    if stats_dict['correct'] != 0:
        prec = float(stats_dict['correct'] / stats_dict['pred'])
        recall = float(stats_dict['correct'] / stats_dict['label'])
        f1 = float(prec * recall) / float(prec + recall) * 2
    metric = {'precision': round(prec, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    return metric


def evaluation_ner_v4(eval_file):
    '''text level strict evaluation
    '''
    with open(eval_file, 'r') as fin:
        lines = fin.readlines()
    
    stats_dict = get_ner_stats(lines)
    print('stats_dict: ', stats_dict)
    
    metric = calculate_ner_metrics(stats_dict)
    print(metric)
    return metric


def get_classify_stats(lines):
    label_stats = defaultdict(int)
    # First pass to get labels
    for line in lines:
        if isinstance(line, str):
            res = json.loads(line.strip())
        else:
            res = line
        label = res['label']
        label_stats[label] += 1
        
    acc_stats = {'label': 0, 'correct': 0} 
    prf_stats = defaultdict(int) 
    
    for line in lines:
        if isinstance(line, str):
            res = json.loads(line.strip())
        else:
            res = line
        label = res['label']
        acc_stats['label'] += 1
        try:
            pred = res['prediction'].split('\n\n### Response: \n')[1].split('\n')[0].strip()
            pred = pred.split(' ')[0].lower()

            if pred in label_stats.keys():
                prf_stats[(label, pred)] += 1
                acc_stats['correct'] += int(pred == label)
            else:
                # print('One item failed, the prediction is ', pred)
                pass
        except:
            pass
            
    return acc_stats, prf_stats, label_stats


def evaluation_classify(eval_file):
    """evluation for classification tasks, assume multiple classes
    """
    with open(eval_file, 'r') as fin:
        lines = fin.readlines()

    acc_stats, prf_stats, label_stats = get_classify_stats(lines)
    print('Label distribution: ', dict(label_stats))
    print(acc_stats)
    # print(len(prf_stats))
    # print(prf_stats)
    
    acc = acc_stats['correct'] / acc_stats['label'] if acc_stats['label'] > 0 else 0
    acc_metric = {'accuracy': round(acc, 4)}
    print(acc_metric)
    macro_metric = calculate_macro_f1(prf_stats)
    print(macro_metric)
    micro_metric = calculate_micro_f1(prf_stats)
    print(micro_metric)
    return


def calculate_macro_f1(prf_stats):
    # Initialize dictionaries to store precision, recall, and F1 for each class
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}

    # Get all unique labels
    labels = set()
    for (true_label, pred_label) in prf_stats.keys():
        labels.add(true_label)

    # Compute precision, recall, and F1 for each class
    for label in labels:
        tp = prf_stats.get((label, label), 0)  # True positives
        fp = sum(prf_stats.get((other_label, label), 0) for other_label in labels if other_label != label)  # False positives
        fn = sum(prf_stats.get((label, other_label), 0) for other_label in labels if other_label != label)  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class[label] = precision
        recall_per_class[label] = recall
        f1_per_class[label] = f1

    # Compute macro F1 by averaging F1 scores across all classes
    macro_precision = sum(precision_per_class.values()) / len(labels)
    macro_recall = sum(recall_per_class.values()) / len(labels)
    macro_f1 = sum(f1_per_class.values()) / len(labels)

    return {
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
    }


def calculate_micro_f1(prf_stats):
    # Initialize counters for true positives, false positives, and false negatives
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Get all unique labels
    labels = set()
    for (true_label, pred_label) in prf_stats.keys():
        labels.add(true_label)

    # Compute total true positives, false positives, and false negatives across all classes
    for label in labels:
        tp = prf_stats.get((label, label), 0)  # True positives
        fp = sum(prf_stats.get((other_label, label), 0) for other_label in labels if other_label != label)  # False positives
        fn = sum(prf_stats.get((label, other_label), 0) for other_label in labels if other_label != label)  # False negatives

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute micro precision, recall, and F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'micro_precision': round(precision, 4),
        'micro_recall': round(recall, 4),
        'micro_f1': round(f1, 4),
    }