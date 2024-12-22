import os
import re

import bert_score
from bert_score import BERTScorer



auto_metrics = ['rouge_1', 'rouge_2', 'rouge_l', 'bert_score']

import numpy as np
#from moverscore_v2 import get_idf_dict, word_mover_score


def get_sents_str(file_path):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub(' ', '', line).strip()
            sents.append(line)
    return sents



def change_word2id_split(ref, pred):
    ref_id, pred_id = [], []
    tmp_dict = {'%': 0}
    new_index = 1
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
        if w == '。':
            ref_id.append(str(0))
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
        if w == '。':
            pred_id.append(str(0))
    return ' '.join(ref_id), ' '.join(pred_id)



def read_rouge_score(name):
    with open(name, 'r') as f:
        lines = f.readlines()
    r1 = lines[3][21:28]
    r2 = lines[7][21:28]
    rl = lines[11][21:28]
    rl_p = lines[10][21:28]
    rl_r = lines[9][21:28]
    return [float(r1), float(r2), float(rl), float(rl_p), float(rl_r)]


def calculate_zh(save_path, pred_file, ref_file, mode):
    refs = get_sents_str(ref_file)
    preds = get_sents_str(pred_file)

    scores = []
    # get rouge scores
    print('Running ROUGE for ' + mode + '-----------------------------', flush=True)
    pred_ids, ref_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id_split(ref, pred)
        pred_ids.append(pred_id)
        ref_ids.append(ref_id)

    with open(os.path.join(save_path, "ref_ids.txt".format(mode)), 'w') as f:
        for ref_id in ref_ids:
            f.write(ref_id + '\n')
    with open(os.path.join(save_path, "pred_ids.txt".format(mode)), 'w') as f:
        for pred_id in pred_ids:
            f.write(pred_id + '\n')
    os.system('files2rouge %s/ref_ids.txt  %s/pred_ids.txt -s %s/rouge_score.txt -e 0' % (
        save_path, save_path, save_path))
    rouge_scores = read_rouge_score(os.path.join(save_path, 'rouge_score.txt'))
    scores.append(rouge_scores[0])
    scores.append(rouge_scores[1])
    scores.append(rouge_scores[2])

    # run bertscore
    print('Running BERTScore for ' + mode + '-----------------------------', flush=True)
    prec, rec, f1 = bert_score.score(preds, refs, lang='zh')
    scores.append(f1.numpy().mean().item())

    score_dict = {}
    for i, metric in enumerate(auto_metrics):
        score_dict[metric] = scores[i]
    print(score_dict, flush=True)
    return rouge_scores[2]


if __name__ == '__main__':

    print("user")

    calculate_zh("./eval_summ/chatgpt2/user_test_results", "./eval_summ/bart_both_csds/user_test_results/hyper.txt",
                 "./eval_summ/bart_both_csds/user_test_results/ref.txt", mode="user")

    print("agent")
    calculate_zh("./eval_summ/bart_both_csds/agent_test_results", "./eval_summ/bart_both_csds/agent_test_results/hyper.txt",
                 "./eval_summ/bart_both_csds/agent_test_results/ref.txt", mode="agent")

    print("final")

    calculate_zh("./eval_summ/chatgpt2/final_test_results",
                 "./eval_summ/chatgpt2/final_test_results/hyper.txt",
                 "./eval_summ/chatgpt2/final_test_results/ref.txt", mode="final")




