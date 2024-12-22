import random

import torch
from logging_utils import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_dataset(data_path, dataset_name, corpus_type):
    assert corpus_type in ["train", "val", "test"]

    pt_file = data_path + dataset_name + '.' + corpus_type + '.pt'
    dataset = torch.load(pt_file)
    logger.info('Loading %s dataset from %s, number of examples: %d' %
                (corpus_type, pt_file, len(dataset)))
    return dataset


def pad(data, pad_id, width=-1):
    if width == -1:
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def make_mask(input_id):
    return [float(x != 0) for x in input_id]


def abs_batch_fn(data, is_train):
    src_ids = [x[0] for x in data]
    role_ids = [x[1] for x in data]
    utt_ids = [x[2] for x in data]
    decoder_input_ids = [x[3] for x in data]
    labels=[x[4] for x in data]
    prompt=[x[5] for x in data]
    tgt_txt = [x[6] for x in data]
    input_ids = pad(src_ids, 0)
    attention_masks = [make_mask(input_id) for input_id in input_ids]
    role_ids = pad(role_ids, 0)
    utt_ids = pad(utt_ids, 0)
    decoder_input_ids = pad(decoder_input_ids, 0)
    labels = pad(labels, -100)
    prompt = pad(prompt, -100)
    input_ids = torch.tensor(input_ids)
    role_ids = torch.tensor(role_ids)
    utt_ids = torch.tensor(utt_ids)
    decoder_input_ids = torch.tensor(decoder_input_ids)
    labels = torch.tensor(labels)
    prompt = torch.tensor(prompt)
    attention_masks = torch.tensor(attention_masks)
    if is_train:
        all_src_ids = [x[7] for x in data]

        all_role_ids = [x[8] for x in data]
        all_utt_ids = [x[9] for x in data]
        contrast_labels = [x[10] for x in data]

        # print("all_utt_ids")
        # print(all_utt_ids)
        # print("all_role_ids")
        # print(all_role_ids)
        new_all_src_ids = []
        new_all_role_ids = []
        new_all_utt_ids = []

        for src_id in all_src_ids:
            for id in src_id:
                new_all_src_ids.append(id)

        for role_id in all_role_ids:
            for id in role_id:
                new_all_role_ids.append(id)
        for utt_id in all_utt_ids:
            for id in utt_id:
                new_all_utt_ids.append(id)
        #print(new_all_src_ids)
        all_input_ids = pad(new_all_src_ids, 0)
        all_attention_masks = [make_mask(input_id) for input_id in all_input_ids]
        all_role_ids = pad(new_all_role_ids, 0)
        all_utt_ids = pad(new_all_utt_ids, 0)
        all_input_ids = torch.tensor(all_input_ids)
        all_role_ids = torch.tensor(all_role_ids)
        all_utt_ids = torch.tensor(all_utt_ids)
        all_attention_masks = torch.tensor(all_attention_masks)
        contrast_labels = torch.tensor(contrast_labels)
        return input_ids, role_ids, utt_ids, attention_masks, decoder_input_ids,labels, tgt_txt, all_input_ids, all_attention_masks, all_role_ids, all_utt_ids, contrast_labels

    else:
        return input_ids, role_ids, utt_ids, attention_masks, tgt_txt,prompt


def build_abs_dataloader(args, datas, batch_size, shuffle, is_train):
    def get_collate_fn(datas):
        return abs_batch_fn(datas, is_train)

    dataset = Bart_Dataset(args, datas,is_train)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=get_collate_fn
    )
    del dataset
    return dataloader


def preprocess_abs_data(args, datas, sample_num,is_train):
    contents = []

    for ex in tqdm(datas):
        if len(ex) == 0:
            continue
        dialogue = ex["dialogue"]
        srcs = dialogue["src_id"]
        role_ids = dialogue["role_ids"]
        utt_ids = dialogue["utt_ids"]
        srcs = srcs[:-1][:args.max_pos - 1] + [srcs[-1]]
        role_ids = role_ids[:-1][:args.max_pos - 1] + [role_ids[-1]]
        utt_ids = utt_ids[:-1][:args.max_pos - 1] + [utt_ids[-1]]
        decoder_input_ids = ex["summary"]["decoder_input_ids"]
        labels = ex["summary"]["labels"]
        prompt = ex["summary"]["start_token"]
        tgt_txt = ex["summary"]["original_txt"]
        if is_train:
            all_srcs = []
            all_role_ids = []
            all_utt_ids = []
            contrast_labels = []
            contrast_labels.append(1)
            contrast_user_sessions = ex["contrast_user_sessions"]["features"]
            contrast_agent_sessions = ex["contrast_agent_sessions"]["features"]
            sample_contrast_user_sessions = random.sample(contrast_user_sessions, sample_num)
            sample_contrast_agent_sessions = random.sample(contrast_agent_sessions, sample_num)
            for neg_dialogue in sample_contrast_user_sessions:
                neg_src_ids = neg_dialogue["src_id"]
                neg_utt_ids = neg_dialogue["utt_ids"]
                neg_role_ids = neg_dialogue["role_ids"]
                neg_src_ids = neg_src_ids[:-1][:args.contrast_max_pos - 1] + [neg_src_ids[-1]]
                neg_role_ids = neg_role_ids[:-1][:args.contrast_max_pos - 1] + [neg_role_ids[-1]]
                neg_utt_ids = neg_utt_ids[:-1][:args.contrast_max_pos - 1] + [neg_utt_ids[-1]]
                all_srcs.append(neg_src_ids)
                all_role_ids.append(neg_role_ids)
                all_utt_ids.append(neg_utt_ids)
                contrast_labels.append(0)
            for neg_dialogue in sample_contrast_agent_sessions:
                neg_src_ids = neg_dialogue["src_id"]
                neg_utt_ids = neg_dialogue["utt_ids"]
                neg_role_ids = neg_dialogue["role_ids"]
                neg_src_ids = neg_src_ids[:-1][:args.contrast_max_pos - 1] + [neg_src_ids[-1]]
                neg_role_ids = neg_role_ids[:-1][:args.contrast_max_pos - 1] + [neg_role_ids[-1]]
                neg_utt_ids = neg_utt_ids[:-1][:args.contrast_max_pos - 1] + [neg_utt_ids[-1]]
                all_srcs.append(neg_src_ids)
                all_role_ids.append(neg_role_ids)
                all_utt_ids.append(neg_utt_ids)
                contrast_labels.append(0)

            assert len(all_srcs) == 2 * sample_num
            assert len(all_role_ids) == 2 * sample_num
            assert len(all_utt_ids) == 2 * sample_num
            contents.append([srcs, role_ids, utt_ids, decoder_input_ids,labels,prompt, tgt_txt, all_srcs, all_role_ids, all_utt_ids, contrast_labels])
        else:
            contents.append(
                [srcs, role_ids, utt_ids, decoder_input_ids, labels, prompt, tgt_txt])

    return contents


class Bart_Dataset(Dataset):
    def __init__(self, args, datas,is_train):
        self.args = args
        self.datas = preprocess_abs_data(args, datas, args.sample_num,is_train)
        self.length = len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return self.length
