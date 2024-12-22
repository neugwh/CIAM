# encoding=utf-8

import argparse
from logging_utils import init_logger

from prepro import data_builder_spe_contrast


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)
    parser.add_argument("-type", default='train', type=str)
    parser.add_argument("-raw_path", default='json_data')
    parser.add_argument("-save_path", default='bert_data')
    parser.add_argument("-bert_dir", default='./bert_chinese')
    parser.add_argument('-min_src_ntokens', default=1, type=int)
    parser.add_argument('-max_src_ntokens', default=3000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=250, type=int)
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument('-min_turns', default=1, type=int)
    parser.add_argument('-max_turns', default=100, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-ex_max_token_num", default=500, type=int)
    parser.add_argument("-truncated", type=str2bool,  default=False)
    parser.add_argument("-add_ex_label", type=str2bool, default=False)
    parser.add_argument('-log_file', default='logs/preprocess.log')
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)
    parser.add_argument("-type", default='train', type=str)
    parser.add_argument("-raw_path", default='json_data')
    parser.add_argument("-save_path", default='bert_data')
    parser.add_argument("-bert_dir", default='./bert_chinese')
    parser.add_argument('-min_src_ntokens', default=1, type=int)
    parser.add_argument('-max_src_ntokens', default=3000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=250, type=int)
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument('-min_turns', default=1, type=int)
    parser.add_argument('-max_turns', default=100, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-ex_max_token_num", default=500, type=int)
    parser.add_argument("-truncated", type=str2bool, default=False)
    parser.add_argument("-add_ex_label", type=str2bool, default=False)
    parser.add_argument('-log_file', default='logs/preprocess.log')
    parser.add_argument('-dataset_name', default='csds')

    args = parser.parse_args()
    if args.type not in ["train", "dev", "test"]:
        print("Invalid data type! Data type should be 'train', 'dev', or 'test'.")
        exit(0)
    init_logger(args.log_file)

    data_builder_spe_contrast.format_to_bert(args)

