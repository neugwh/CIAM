import re
import files2rouge
import os
def get_sents_str(f):
    sents = []
    for line in f.readlines():
        line = re.sub(' ', '', line.strip())
        sents.append(line)
    return sents
def read_rouge_score(name):
    with open(name, 'r') as f:
        lines = f.readlines()
    r1 = lines[3][21:28]
    r2 = lines[7][21:28]
    rl = lines[11][21:28]
    return [float(r1), float(r2), float(rl)]
def change_word2id(ref, pred):
    ref = re.sub(' ', '', ref)
    pred = re.sub(' ', '', pred)
    pred = re.sub('<q>', '', pred)
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


def cal_rouge(pred_name, ref_name):
    refs = get_sents_str(ref_name)
    preds = get_sents_str(pred_name)
    # write ids
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id(ref, pred)
        ref_ids.append(ref_id)
        pred_ids.append(pred_id)
    with open('logs/ref_ids.txt', 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open('logs/pred_ids.txt', 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    files2rouge.run('logs/pred_ids.txt', 'logs/ref_ids.txt')

def cal_rouge_path(save_path,pred_name, ref_name):
    with open(ref_name, 'r') as f:
        refs = get_sents_str(f)
    with open(pred_name, 'r') as f:
        preds = get_sents_str(f)
    # write ids
    ref_ids, pred_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id(ref, pred)
        ref_ids.append(ref_id)
        pred_ids.append(pred_id)
    with open(os.path.join(save_path,"ref_ids.txt"), 'w') as f:
        for ref in ref_ids:
            f.write(ref + '\n')
    with open(os.path.join(save_path,"pred_ids.txt"), 'w') as f:
        for pred in pred_ids:
            f.write(pred + '\n')
    print("Running ROUGE", flush=True)
    os.system('files2rouge %s/ref_ids.txt  %s/pred_ids.txt -s %s/rouge_score.txt -e 0' % (
        save_path, save_path, save_path))
    rouge_scores = read_rouge_score(os.path.join(save_path, 'rouge_score.txt'))
    print('ROUGE-1-F: ', rouge_scores[0], flush=True)
    print('ROUGE-2-F: ', rouge_scores[1], flush=True)
    print('ROUGE-L-F: ', rouge_scores[2], flush=True)