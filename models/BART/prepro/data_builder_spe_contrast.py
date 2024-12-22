import gc
import glob
import json
from os.path import join as pjoin

import torch
from logging_utils import logger
from transformers import BertTokenizer


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.eou_token = '[EOU]'
        self.eop_token = '[EOP]'
        self.user_bos_token='[USERSUM]'
        self.agent_bos_token='[AGENTSUM]'
        self.final_bos_token='[FINALSUM]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]
        self.eou_vid = self.tokenizer.vocab[self.eou_token]
        self.eop_vid = self.tokenizer.vocab[self.eop_token]

        self.user_bos_vid=self.tokenizer.vocab[self.user_bos_token]
        self.agent_bos_vid=self.tokenizer.vocab[self.agent_bos_token]
        self.final_bos_vid=self.tokenizer.vocab[self.final_bos_token]
        self.eos_vid = 102

        # self.eos_vid = 102
        # self.bos_vid = 101

    def preprocess_src(self, content, info=None):
        if info == 'agent':
            role = 3
        else:
            role = 2
        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if self.args.truncated:
            content = content[:self.args.max_src_ntokens_per_sent]
        content_subtokens = self.tokenizer.tokenize(content)
        src_subtokens = content_subtokens
        return src_subtokens, role

    def preprocess_summary(self, summary_list,type):
        original_txt = ''.join(summary_list)
        content_subtokens = []
        id = 0
        utt_ids = []

        for summ in summary_list:
            id = id + 1
            sub_tokens = self.tokenizer.tokenize(summ)
            content_subtokens.extend(sub_tokens)
            if (id!=len(summary_list)):
                content_subtokens.append(self.eou_token)
                utt_ids.extend([id] * (len(sub_tokens) + 1))
            else:
                utt_ids.extend([id] * (len(sub_tokens)))
        if type==0:
            decoder_input_ids=[self.user_bos_vid]+self.tokenizer.convert_tokens_to_ids(content_subtokens)
            start_token=[self.user_bos_vid]
        elif type==1:
            decoder_input_ids = [self.agent_bos_vid] + self.tokenizer.convert_tokens_to_ids(content_subtokens)
            start_token = [self.agent_bos_vid]
        elif type==2:
            decoder_input_ids = [self.final_bos_vid] + self.tokenizer.convert_tokens_to_ids(content_subtokens)
            start_token = [self.final_bos_vid]

        labels = self.tokenizer.convert_tokens_to_ids(content_subtokens) + [self.eos_vid]
        utt_ids = [utt_ids[0]] + utt_ids
        assert len(decoder_input_ids) == len(utt_ids)
        return decoder_input_ids,labels,start_token, original_txt, utt_ids

    def integrate_dialogue(self, dialogue):
        src_tokens = []
        role_ids = []
        id = 0
        utt_ids = []
        cls_ids = []
        roles = []
        seg_ids = []
        for sent in dialogue:
            id = id + 1
            tokens = sent["src_tokens"]
            cls_ids.append(len(src_tokens))
            src_tokens.extend(tokens)
            role = sent["role"]
            src_tokens.append(self.eou_token)
            role_ids.extend([role] * (len(tokens) + 1))
            utt_ids.extend([id] * (len(tokens) + 1))
            roles.append(role)
            if id % 2:
                seg_ids.extend([0] * (len(tokens) + 1))
            else:
                seg_ids.extend([1] * (len(tokens) + 1))

        src_ids = [self.cls_vid] + self.tokenizer.convert_tokens_to_ids(src_tokens) + [self.sep_vid]
        # src_ids = [self.bos_vid] + self.tokenizer.convert_tokens_to_ids(src_tokens)[:-1] + [self.eos_vid]
        role_ids = [role_ids[0]] + role_ids + [role_ids[-1]]
        utt_ids = [utt_ids[0]] + utt_ids + [utt_ids[-1]]
        seg_ids = [seg_ids[0]] + seg_ids + [seg_ids[-1]]
        assert len(cls_ids) == len(roles)
        assert len(role_ids) == len(src_ids)
        assert len(role_ids) == len(utt_ids)
        assert len(seg_ids) == len(role_ids)
        return {"src_id": src_ids, "role_ids": role_ids, "utt_ids": utt_ids, "seg_ids": seg_ids}

def format_to_bert(args, corpus_type=None):
    a_lst = []

    if corpus_type is not None:
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')):
            real_name = json_f.split('/')[-1]

            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))
    else:
        for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
            real_name = json_f.split('/')[-1]

            corpus_type = real_name.split('.')[1]
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))

    for d in a_lst:
        statistic = _format_to_bert(d)
        if statistic is None:
            continue


def convert_dialogue(session, bert):
    dialogue_b_data = []
    for index, sent in enumerate(session):
        content = sent['content']
        role = sent['type']
        b_data = bert.preprocess_src(content, role)
        if (b_data is None):
            continue
        src_subtokens, roles = b_data
        b_data_dict = {"src_tokens": src_subtokens, "role": roles}
        dialogue_b_data.append(b_data_dict)
    return dialogue_b_data


def convert_contrast_session(expand_session, bert):
    features = []
    for session in expand_session:
        dialogue_b_data = convert_dialogue(session, bert)
        dialogue_integrated_pad = bert.integrate_dialogue(dialogue_b_data)
        features.append(dialogue_integrated_pad)


    return {"features": features}


def _format_to_bert(params):
    _, json_file, args, save_file = params

    bert = BertData(args)
    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file, encoding="utf-8"))

    datasets = []

    count = 0

    for dialogue in jobs:
        type=dialogue['type']
        dialogue_b_data = convert_dialogue(dialogue["session"], bert)
        dialogue_example = {"session": dialogue_b_data}
        dialogue_integrated = bert.integrate_dialogue(dialogue_b_data)
        dialogue_example["dialogue"] = dialogue_integrated
        summary = dialogue['summary']
        summ_b_data = bert.preprocess_summary(summary,type)
        decoder_input_ids, labels, satrt_token, original_txt, utt_ids = summ_b_data
        b_data_dict = {"decoder_input_ids": decoder_input_ids, "labels": labels, "start_token": satrt_token,
                       "original_txt": original_txt, "utt_ids": utt_ids}
        dialogue_example["summary"] = b_data_dict
        if 'contrast_user_sessions' in dialogue.keys() and 'contrast_agent_sessions' in dialogue.keys():
            dialogue_contrast_user_sessions = convert_contrast_session(dialogue['contrast_user_sessions'], bert)
            dialogue_contrast_agent_sessions = convert_contrast_session(dialogue['contrast_agent_sessions'],
                                                                        bert)
            dialogue_example["contrast_user_sessions"] = dialogue_contrast_user_sessions
            dialogue_example["contrast_agent_sessions"] = dialogue_contrast_agent_sessions
        if len(dialogue_b_data) >= args.min_turns:
            datasets.append(dialogue_example)
            count += 1
            if count % 50 == 0:
                print(count)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
