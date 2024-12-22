import bisect
import gc
import glob
import random

import torch

from logging_utils import logger
from pathlib import Path


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_train=False):
        """Create a Batch from a list of examples."""

        if data is not None:
            self.batch_size = len(data)
            pre_src_ids = [x[0] for x in data]
            pre_role_ids = [x[1] for x in data]
            pre_utt_ids = [x[2] for x in data]
            pre_seg_ids = [x[3] for x in data]
            pre_tgt = [x[4] for x in data]
            pre_prompt = [x[5] for x in data]
            tgt_txt = [x[6] for x in data]
            src_ids = torch.tensor(self._pad(pre_src_ids, 0))
            seg_ids = torch.tensor(self._pad(pre_seg_ids, 0))
            role_ids = torch.tensor(self._pad(pre_role_ids, 0))
            utt_ids = torch.tensor(self._pad(pre_utt_ids, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))
            prompt = torch.tensor(self._pad(pre_prompt, 0))

            mask_src = ~(src_ids == 0)
            attention_mask = (src_ids!=0).float()
            mask_tgt = ~(tgt == 0)

            setattr(self, 'src', src_ids.to(device))
            setattr(self, 'seg_ids', seg_ids.to(device))
            setattr(self, 'role_ids', role_ids.to(device))
            setattr(self, 'utt_ids', utt_ids.to(device))
            setattr(self, 'tgt', tgt.to(device))


            setattr(self, 'prompt', prompt.to(device))
            setattr(self, 'tgt_txt', tgt_txt)
            setattr(self, 'attention_mask',attention_mask.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if (is_train):
                all_src_ids = [x[7] for x in data]
                all_role_ids = [x[8] for x in data]
                all_utt_ids = [x[9] for x in data]
                all_seg_ids = [x[10] for x in data]
                contrast_labels = [x[11] for x in data]
                new_all_src_ids = []
                new_all_role_ids = []
                new_all_utt_ids = []
                new_all_seg_ids = []
                for src_id in all_src_ids:
                    for id in src_id:
                        new_all_src_ids.append(id)

                for role_id in all_role_ids:
                    for id in role_id:
                        new_all_role_ids.append(id)
                for utt_id in all_utt_ids:
                    for id in utt_id:
                        new_all_utt_ids.append(id)
                for seg_id in all_seg_ids:
                    for id in seg_id:
                        new_all_seg_ids.append(id)
                all_src_ids = torch.tensor(self._pad(new_all_src_ids, 0))
                all_seg_ids = torch.tensor(self._pad(new_all_seg_ids, 0))
                all_role_ids = torch.tensor(self._pad(new_all_role_ids, 0))
                all_utt_ids = torch.tensor(self._pad(new_all_utt_ids, 0))
                contrast_labels = torch.tensor(contrast_labels)

                all_mask_src=(all_src_ids != 0).float()
                all_attention_mask = all_mask_src.float()
                setattr(self, 'all_src', all_src_ids.to(device))
                setattr(self, 'all_seg_ids', all_seg_ids.to(device))
                setattr(self, 'all_role_ids', all_role_ids.to(device))
                setattr(self, 'all_utt_ids', all_utt_ids.to(device))
                setattr(self, 'all_attention_mask', all_attention_mask.to(device))
                setattr(self, 'all_mask_src', all_mask_src.to(device))
                setattr(self, 'contrast_labels', contrast_labels.to(device))

    def __len__(self):
        return self.batch_size


def load_dataset(data_path, dataset_name, corpus_type):
    assert corpus_type in ["train", "val", "test"]

    def _lazy_dataset_loader(data_path, dataset_name, corpus_type):
        pt_file = data_path + dataset_name + '.' + corpus_type + '.pt'
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    yield _lazy_dataset_loader(data_path, dataset_name, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    # print(count, src_elements)
    # if (count > 6):
    #     return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_train):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)

        except StopIteration:
            return None

        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_train=self.is_train)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_train=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.dataset = batch_size, dataset
        self.is_train = is_train
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset

        return xs

    def preprocess(self, ex, is_train):

        dialogue = ex["dialogue"]
        src_ids = dialogue["src_id"]
        role_ids = dialogue["role_ids"]
        utt_ids = dialogue["utt_ids"]
        seg_ids = dialogue["seg_ids"]
        all_src_ids = []
        all_role_ids = []
        all_utt_ids = []
        all_seg_ids = []
        contrast_labels = []
        contrast_labels.append(1)
        src_ids = src_ids[:-1][:self.args.max_pos - 1] + [src_ids[-1]]
        role_ids = role_ids[:-1][:self.args.max_pos - 1] + [role_ids[-1]]
        utt_ids = utt_ids[:-1][:self.args.max_pos - 1] + [utt_ids[-1]]
        seg_ids = seg_ids[:-1][:self.args.max_pos - 1] + [seg_ids[-1]]

        tgt=ex["summary"]["tgt"]
        prompt = ex["summary"]["start_token"]
        tgt_txt = ex["summary"]["original_txt"]
        if is_train:
            contrast_user_sessions = ex["contrast_user_sessions"]["features"]
            contrast_agent_sessions = ex["contrast_agent_sessions"]["features"]
            sample_contrast_user_sessions = random.sample(contrast_user_sessions, self.args.sample_num)
            sample_contrast_agent_sessions = random.sample(contrast_agent_sessions, self.args.sample_num)
            for neg_dialogue in sample_contrast_user_sessions:
                neg_src_ids = neg_dialogue["src_id"]
                neg_utt_ids = neg_dialogue["utt_ids"]
                neg_role_ids = neg_dialogue["role_ids"]
                neg_seg_ids = neg_dialogue["seg_ids"]
                neg_src_ids = neg_src_ids[:-1][:self.args.max_pos - 1] + [neg_src_ids[-1]]
                neg_role_ids = neg_role_ids[:-1][:self.args.max_pos - 1] + [neg_role_ids[-1]]
                neg_utt_ids = neg_utt_ids[:-1][:self.args.max_pos - 1] + [neg_utt_ids[-1]]
                neg_seg_ids = neg_seg_ids[:-1][:self.args.max_pos - 1] + [neg_seg_ids[-1]]
                all_src_ids.append(neg_src_ids)
                all_role_ids.append(neg_role_ids)
                all_utt_ids.append(neg_utt_ids)
                all_seg_ids.append(neg_seg_ids)
                contrast_labels.append(0)
            for neg_dialogue in sample_contrast_agent_sessions:
                neg_src_ids = neg_dialogue["src_id"]
                neg_utt_ids = neg_dialogue["utt_ids"]
                neg_role_ids = neg_dialogue["role_ids"]
                neg_seg_ids = neg_dialogue["seg_ids"]
                neg_src_ids = neg_src_ids[:-1][:self.args.max_pos - 1] + [neg_src_ids[-1]]
                neg_role_ids = neg_role_ids[:-1][:self.args.max_pos - 1] + [neg_role_ids[-1]]
                neg_utt_ids = neg_utt_ids[:-1][:self.args.max_pos - 1] + [neg_utt_ids[-1]]
                neg_seg_ids = neg_seg_ids[:-1][:self.args.max_pos - 1] + [neg_seg_ids[-1]]
                all_src_ids.append(neg_src_ids)
                all_role_ids.append(neg_role_ids)
                all_utt_ids.append(neg_utt_ids)
                all_seg_ids.append(neg_seg_ids)
                contrast_labels.append(0)


            assert len(all_src_ids) == 2 * self.args.sample_num
            assert len(all_role_ids) == 2 * self.args.sample_num
            assert len(all_utt_ids) == 2 * self.args.sample_num
            assert len(all_seg_ids) == 2 * self.args.sample_num

            return src_ids, role_ids, utt_ids, seg_ids, tgt,prompt, tgt_txt, all_src_ids, all_role_ids, all_utt_ids, all_seg_ids, contrast_labels
        else:
            return src_ids, role_ids, utt_ids, seg_ids, tgt, prompt, tgt_txt

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['dialogue']['src_id']) == 0):
                continue
            ex = self.preprocess(ex, self.is_train)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far >= batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            # elif size_so_far > batch_size:
            #     yield minibatch[:-1]
            #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            # elif size_so_far > batch_size:
            #     yield minibatch[:-1]
            #     minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 200):

            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_train)

                yield batch
            return
