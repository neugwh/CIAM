#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import math
import os
import random
import numpy as np
import torch
from data_loader_spe_contrast import build_abs_dataloader
from data_loader_spe_contrast import load_dataset
from logging_utils import init_logger
from logging_utils import logger
from network_softmax_aware import ContrastSummModel
#from network_softmax_wl import ContrastSummModel

#from optimizers import build_optim_bart
from optimizers import build_optim_bart2
from others.evaluate import calculate_zh
from tqdm import tqdm
from transformers import BartForConditionalGeneration,set_seed
from transformers import BertTokenizer, get_scheduler, BartTokenizer


# from network import ContrastSummModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'val', 'test'])
    parser.add_argument("-dataset_name", default='csds', type=str, choices=['csds', 'mc'])
    parser.add_argument("-data_path", default='bert_data_spe_contrast/all/')
    parser.add_argument("-user_data_path", default='bert_data_spe_contrast/user/')
    parser.add_argument("-agent_data_path", default='bert_data_spe_contrast/agent/')
    parser.add_argument("-final_data_path", default='bert_data_spe_contrast/final/')
    parser.add_argument("--tokenizer_name", type=str, default="./bert_chinese",
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--model_name", default="./bart_base_chinese", type=str, help="")
    parser.add_argument("--max_pos", default=512, type=int, help="")
    parser.add_argument("--hidden_size", default=768, type=int, help="")
    parser.add_argument("--batch_size", default=4, type=int, help="")
    parser.add_argument("--test_batch_size", default=16, type=int, help="")
    parser.add_argument("--lr", default=3e-5, type=float, help="")
    parser.add_argument("--train_epochs", default=5, type=int, help="")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--contrast_max_pos", default=512, type=int, help="")
    parser.add_argument("--sample_num", default=3, type=int, help="")
    parser.add_argument("--max_turn_range", default=8, type=int, help="")
    parser.add_argument("--temperature", default=0.1, type=float, help="")
    parser.add_argument("--role_lambda", default=0.4, type=float, help="")

    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"], )
    parser.add_argument("--warmup_steps", type=float, default=800,
                        help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--vocab_path", default=None, type=str, help="")
    parser.add_argument("--smooth_label", default=False, type=bool, help="")
    parser.add_argument("--ext_label", default=False, type=bool, help="")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="checkpoint_path")
    parser.add_argument("--max_tgt_len", type=int, default=150)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--min_tgt_len", type=int, default=10)

    parser.add_argument("-model_path", default='models/bart_base_user')
    parser.add_argument("-val_result_path", default='val_results')
    parser.add_argument("-user_val_result_path", default='user_val_results')
    parser.add_argument("-agent_val_result_path", default='agent_val_results')
    parser.add_argument("-final_val_result_path", default='final_val_results')
    parser.add_argument("-user_test_result_path", default='user_test_results')
    parser.add_argument("-agent_test_result_path", default='agent_test_results')
    parser.add_argument("-final_test_result_path", default='final_test_results')

    parser.add_argument("-test_result_path", default='test_results')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-log_file', default='./logs/train_bart_src.log')
    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-device_id', default=0, type=int)

    # Batch sizes
    # Training process args
    parser.add_argument("-do_eval", type=bool, default=True)
    parser.add_argument("-save_every", default=1, type=int)
    parser.add_argument("-eval_every", default=1, type=int)
    parser.add_argument("--eval_every_step", default=9000, type=int)
    parser.add_argument("-print_every", default=500, type=int)

    parser.add_argument("-train_steps", default=None, type=int)
    parser.add_argument("-max_grad_norm", default=1.0, type=float)
    args = parser.parse_args()
    return args


def save_model(args, model, epoch):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save({
        "model": model.state_dict(),
    }, os.path.join(args.model_path, "abs_bart_base{}.tar".format(epoch)))


def validate_abs(args, model, validate_loader, tokenizer, epoch, device, mode):
    logger.info("***** Running valid *****")
    model.eval()
    logger.info("Evaluate:: Epoch: {0}".format(epoch))

    total_loss = 0.0
    hypothesis = []
    references = []

    for step, batch in enumerate(validate_loader):
        input_ids, role_ids, utt_ids, attention_masks, tgt_txt, prompt = batch
        utt_ids = utt_ids.to(device)
        input_ids = input_ids.to(device)
        role_ids = role_ids.to(device)
        attention_mask = attention_masks.to(device)
        prompt = prompt.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_ids,
                decoder_input_ids=prompt,
                attention_mask=attention_mask,
            )
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            preds = [pred.strip() for pred in decoded_preds]
            refs = [tgt.strip() for tgt in tgt_txt]
            preds = [pred.replace(" ", "") for pred in preds]
            refs = [ref.replace(" ", "") for ref in refs]
            preds = [pred.replace("[UNK]", "") for pred in preds]
            preds = [pred.replace("[EOU]", "") for pred in preds]
            preds = [pred.replace("[EOP]", "") for pred in preds]
            preds = [pred.replace("[U]", "") for pred in preds]
            preds = [pred.replace("[A]", "") for pred in preds]
            preds = [pred.replace("[USERSUM]", "") for pred in preds]
            preds = [pred.replace("[AGENTSUM]", "") for pred in preds]
            preds = [pred.replace("[FINALSUM]", "") for pred in preds]
            hypothesis.extend(preds)
            references.extend(refs)

    assert len(hypothesis) == len(references)
    result_path = os.path.join(args.model_path, args.val_result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    hyp_path = os.path.join(result_path, "hyper.txt")
    ref_path = os.path.join(result_path, "ref.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in range(len(hypothesis)):
            if i != len(hypothesis) - 1:
                f.write(hypothesis[i] + "\n")
            else:
                f.write(hypothesis[i])
    with open(ref_path, "w", encoding="utf-8") as f:
        for i in range(len(references)):
            if i != len(references) - 1:
                f.write(references[i] + "\n")
            else:
                f.write(references[i])

    rouge_l = calculate_zh(result_path, hyp_path, ref_path, mode)


    return rouge_l


def train_abs(args, model, train_loader, valid_loader, tokenizer, device):
    optimizer = build_optim_bart2(args, model)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    if args.train_steps is None:
        args.train_steps = args.train_epochs * num_update_steps_per_epoch
    else:
        args.train_epochs = math.ceil(args.train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {args.train_epochs}")
    logger.info(f"  Total optimization steps = {args.train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.train_steps))
    completed_steps = 0

    min_loss = 1000000
    max_epoch = 0
    max_user_score = 0.0
    max_agent_score = 0.0
    min_loss = 1000000
    max_score=0.0
    max_epoch = 0
    max_user_step = 0
    max_agent_step = 0
    for epoch in range(args.train_epochs):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, role_ids, utt_ids, attention_masks, decoder_input_ids, labels, tgt_txt, all_input_ids, all_attention_masks, all_role_ids, all_utt_ids, contrast_labels = batch
            input_ids = input_ids.to(device)
            role_ids = role_ids.to(device)
            utt_ids = utt_ids.to(device)
            attention_mask = attention_masks.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            labels = labels.to(device)
            all_input_ids = all_input_ids.to(device)
            all_attention_masks = all_attention_masks.to(device)
            all_role_ids = all_role_ids.to(device)
            all_utt_ids = all_utt_ids.to(device)
            contrast_labels = contrast_labels.to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, \
                         role_ids=role_ids, utt_ids=utt_ids, decoder_input_ids=decoder_input_ids, \
                         all_input_ids=all_input_ids, all_attention_mask=all_attention_masks,
                         all_role_ids=all_role_ids, all_utt_ids=all_utt_ids, contrast_labels=contrast_labels)
            loss = loss / args.accum_steps
            loss.backward()
            total_loss = total_loss + loss.item()
            if step % args.accum_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if completed_steps % args.print_every == 0:
                avg_train_loss = total_loss / args.print_every
                logger.info("completed step{},Average train loss: {}".format(step, avg_train_loss))
                total_loss = 0
        if epoch % args.eval_every == 0 and args.do_eval:
            #if epoch % args.eval_every == 0 and args.do_eval:
            logger.info("*****{}*****".format(epoch))
            logger.info("score")
            rouge_l = test_abs(args, model, valid_loader, tokenizer, args.val_result_path, device,
                                        "all")
            if rouge_l > max_score:
                max_score = rouge_l
                max_epoch= epoch
            logger.info(max_epoch)
            save_model(args, model, epoch)




def test_abs(args, model, test_loader, tokenizer, test_result_path, device, mode):
    logger.info("***** Running test *****")
    model.eval()

    hypothesis = []
    references = []

    for step, batch in enumerate(test_loader):
        input_ids, role_ids, utt_ids, attention_masks, tgt_txt, prompt = batch
        utt_ids = utt_ids.to(device)
        input_ids = input_ids.to(device)
        role_ids = role_ids.to(device)
        attention_mask = attention_masks.to(device)
        prompt = prompt.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_ids,
                decoder_input_ids=prompt,
                attention_mask=attention_mask,
            )

            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            preds = [pred.strip() for pred in decoded_preds]
            refs = [tgt.strip() for tgt in tgt_txt]
            preds = [pred.replace(" ", "") for pred in preds]
            refs = [ref.replace(" ", "") for ref in refs]
            preds = [pred.replace("[UNK]", "") for pred in preds]
            preds = [pred.replace("[EOU]", "") for pred in preds]
            preds = [pred.replace("[EOP]", "") for pred in preds]
            preds = [pred.replace("[U]", "") for pred in preds]
            preds = [pred.replace("[A]", "") for pred in preds]
            preds = [pred.replace("[USERSUM]", "") for pred in preds]
            preds = [pred.replace("[AGENTSUM]", "") for pred in preds]
            preds = [pred.replace("[FINALSUM]", "") for pred in preds]
            hypothesis.extend(preds)
            references.extend(refs)

    assert len(hypothesis) == len(references)
    result_path = os.path.join(args.model_path, test_result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    hyp_path = os.path.join(result_path, "hyper.txt")
    ref_path = os.path.join(result_path, "ref.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in range(len(hypothesis)):
            if i != len(hypothesis) - 1:
                f.write(hypothesis[i] + "\n")
            else:
                f.write(hypothesis[i])
    with open(ref_path, "w", encoding="utf-8") as f:
        for i in range(len(references)):
            if i != len(references) - 1:
                f.write(references[i] + "\n")
            else:
                f.write(references[i])

    rouge_l = calculate_zh(result_path, hyp_path, ref_path, mode)

    return rouge_l


if __name__ == "__main__":
    args = get_args()
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    print(torch.cuda.is_available())
    logger.info('Device ID %d' % args.device_id)
    logger.info('Device %s' % device)
    print(args.batch_size)
    if args.device_id >= 0:
        torch.cuda.set_device(args.device_id)

    set_seed(args.seed)
    bart_model = BartForConditionalGeneration.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
    )

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    bart_model.resize_token_embeddings(len(tokenizer))
    # bart_model = bart_model.to(device)
    contrast_model = ContrastSummModel(bart_model, args, tokenizer)
    contrast_model = contrast_model.to(device)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path,map_location=device)
        contrast_model.load_state_dict(checkpoint['model'], strict=False)
    train_loader = build_abs_dataloader(args,  load_dataset(args.data_path, args.dataset_name, 'train'), args.batch_size,
                                        shuffle=True, is_train=True)


    validate_loader = build_abs_dataloader(args, load_dataset(args.data_path, args.dataset_name, 'val'),
                                           args.test_batch_size,
                                           shuffle=False, is_train=False)
    user_test_loader = build_abs_dataloader(args, load_dataset(args.user_data_path, args.dataset_name, 'test'),
                                            args.test_batch_size,
                                            shuffle=False, is_train=False)
    agent_test_loader = build_abs_dataloader(args, load_dataset(args.agent_data_path, args.dataset_name, 'test'),
                                             args.test_batch_size,
                                             shuffle=False, is_train=False)
    final_test_loader = build_abs_dataloader(args, load_dataset(args.final_data_path, args.dataset_name, 'test'),
                                             args.test_batch_size,
                                             shuffle=False, is_train=False)

    if args.mode == 'train':
        train_abs(args, contrast_model, train_loader, validate_loader, tokenizer,
                  device)

    elif args.mode == 'val':
        validate_abs(args, contrast_model, validate_loader, tokenizer, 0, device, "all")
    elif args.mode == 'test':
        print("user_rouge")
        test_abs(args, contrast_model, user_test_loader, tokenizer, args.user_test_result_path, device, "user")
        print("agent_rouge")
        test_abs(args, contrast_model, agent_test_loader, tokenizer, args.agent_test_result_path, device, "agent")
        print("final_rouge")
        test_abs(args, contrast_model, final_test_loader, tokenizer, args.final_test_result_path, device, "final")
