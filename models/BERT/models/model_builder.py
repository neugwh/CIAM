import copy

import torch
import torch.nn as nn
from models.decoder import TransformerDecoder
from models.optimizers import Optimizer
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('model.bert')]
    optim.set_parameters(params)

    return optim


def build_optim_fuse(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, 0.02, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=500)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('model')]
    optim.set_parameters(params)
    return optim


def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if
              ((not n.startswith('model.bert')) and (n.startswith('model')))]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class FuseLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mid_gate = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, input1, input2):
        mid_query = torch.cat([input1, input2], dim=-1)
        mid = self.mid_gate(mid_query)

        return mid


class AVG(nn.Module):
    """
    对BERT输出的embedding求masked average
    """

    def __init__(self, eps=1e-12):
        super(AVG, self).__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (torch.sum(m, dim=1, keepdims=True) + self.eps)
        avg_output = reduce_mean(hidden_states, attention_mask)
        return avg_output


class Bert(nn.Module):
    def __init__(self, finetune=True):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('./bert_chinese')
        self.finetune = finetune

    def forward(self, x, seg_ids, mask):
        if (self.finetune):
            top_vec = self.model(x, token_type_ids=seg_ids, attention_mask=mask)[0]
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(x, token_type_ids=seg_ids, attention_mask=mask)[0]
        return top_vec


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
            self.bert.model.embeddings.position_ids = torch.arange(args.max_pos).expand((1, -1))
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, seg_ids, mask_src):
        top_vec = self.bert(src, seg_ids, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, top_vec, None

class ContrastAbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(ContrastAbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AbsSummarizer(args, device, bert_from_extractive=bert_from_extractive)
        self.sample_num = 2 * args.sample_num
        self.temperature = args.temperature
        self.max_turn_range = args.max_turn_range
        self.temperature = args.temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = AVG(eps=1e-6)
        self.hidden_size = args.hidden_size
        self.embed_scale = self.hidden_size ** -0.5
        self.user_fuse_layer = FuseLayer(self.hidden_size)
        self.agent_fuse_layer = FuseLayer(self.hidden_size)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.model.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.model.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
        self.to(device)

    def calc_cos(self, x, y):
        """
        计算cosine相似度
        """
        cos = torch.cosine_similarity(x, y, dim=1)
        cos = cos / self.temperature  # cos = cos / 2.0
        return cos

    def calc_loss(self, pred, labels):
        """
        计算损失函数
        """
        # pred = pred.float()
        loss = -torch.mean(self.log_softmax(pred) * labels)
        return loss

    @staticmethod
    def masked_softmax(L, cross_mask, self_mask, view_range_mask):
        '''
        :param L: batch size, n, m
        :param sequence_length: batch size
        :return:
        '''
        bz, n, m = L.shape
        expand_cross_mask = cross_mask[:, None, :].expand(bz, n, m)
        expand_self_mask = self_mask[:, :, None].expand(bz, n, m)
        filterd_expand_mask = expand_cross_mask * view_range_mask * expand_self_mask
        inverted_mask = (1.0 - filterd_expand_mask) * torch.finfo(L.dtype).min
        attention_weight = torch.softmax(L + inverted_mask, dim=-1)
        return attention_weight

    def get_rep(self, encoder_hidden_states, role_ids, utt_ids, attention_mask):
        bsz, src_len, hidden_size = encoder_hidden_states.size()
        one_mask = torch.ones_like(role_ids)
        zero_mask = torch.zeros_like(role_ids)
        role_user_mask = torch.where(role_ids == 2, one_mask, zero_mask)
        role_agent_mask = torch.where(role_ids == 3, one_mask, zero_mask)
        role_user_attention_mask = (attention_mask * role_user_mask)
        role_agent_attention_mask = (attention_mask * role_agent_mask)
        user_self_out = encoder_hidden_states * role_user_attention_mask.unsqueeze(-1)
        agent_self_out = encoder_hidden_states * role_agent_attention_mask.unsqueeze(-1)
        w = torch.matmul((user_self_out * self.embed_scale), agent_self_out.transpose(-1, -2))

        # w = torch.matmul(self.wl(user_self_out)* self.embed_scale, agent_self_out.transpose(-1, -2))

        # w = torch.matmul(user_self_out, agent_self_out.transpose(-1, -2))
        view_turn_mask = utt_ids.unsqueeze(1).repeat(1, src_len, 1)
        view_turn_mask_transpose = view_turn_mask.transpose(2, 1)
        view_range_mask = torch.where(
            abs(view_turn_mask_transpose - view_turn_mask) <= self.max_turn_range,
            torch.ones_like(view_turn_mask),
            torch.zeros_like(view_turn_mask))
        filtered_w = self.masked_softmax(w, role_agent_attention_mask, role_user_attention_mask, view_range_mask)
        inverse_filter_w = self.masked_softmax(w.permute(0, 2, 1), role_user_attention_mask, role_agent_attention_mask,
                                               view_range_mask)
        user_cross_out = torch.matmul(inverse_filter_w, user_self_out)
        agent_cross_out = torch.matmul(filtered_w, agent_self_out)
        user_aware_out = self.user_fuse_layer(user_self_out, agent_cross_out)
        agent_aware_out = self.agent_fuse_layer(agent_self_out, user_cross_out)

        # user_cross_out = torch.matmul(w.permute(0, 2, 1), user_self_out)
        # agent_cross_out = torch.matmul(w, agent_self_out)
        user_self_out = self.avg(user_self_out, role_user_attention_mask)
        user_aware_out = self.avg(user_aware_out, role_user_attention_mask)
        agent_self_out = self.avg(agent_self_out, role_agent_attention_mask)
        agent_aware_out = self.avg(agent_aware_out, role_agent_attention_mask)
        return user_self_out, user_aware_out, agent_self_out, agent_aware_out

    def forward(self, src, tgt, seg_ids, role_ids, utt_ids, mask_src,
                attention_mask=None,
                all_src=None,
                all_seg_ids=None,
                all_role_ids=None,
                all_utt_ids=None,
                all_mask_src=None,
                all_attention_mask=None,
                contrast_labels=None
                ):

        decoder_outputs, pos_encoder_hidden_states, _ = self.model(src, tgt, seg_ids, mask_src)
        logit_user = None
        logit_agent = None

        if contrast_labels is not None and all_src is not None:
            neg_encoder_hidden_states = self.model.bert(
                all_src, all_seg_ids, all_mask_src
            )
            pos_user_self_out, pos_user_aware_out, pos_agent_self_out, pos_agent_aware_out = \
                self.get_rep(pos_encoder_hidden_states, role_ids, utt_ids, attention_mask)
            neg_user_self_out, neg_user_aware_out, neg_agent_self_out, neg_agent_aware_out = \
                self.get_rep(neg_encoder_hidden_states, all_role_ids, all_utt_ids, all_attention_mask)
            pos_user_self_out = pos_user_self_out.view(-1, 1, self.hidden_size)
            pos_user_aware_out = pos_user_aware_out.view(-1, 1, self.hidden_size)
            pos_agent_self_out = pos_agent_self_out.view(-1, 1, self.hidden_size)
            pos_agent_aware_out = pos_agent_aware_out.view(-1, 1, self.hidden_size)

            neg_user_self_out = neg_user_self_out.view(-1, self.sample_num, self.hidden_size)
            neg_user_aware_out = neg_user_aware_out.view(-1, self.sample_num, self.hidden_size)
            neg_agent_self_out = neg_agent_self_out.view(-1, self.sample_num, self.hidden_size)
            neg_agent_aware_out = neg_agent_aware_out.view(-1, self.sample_num, self.hidden_size)
            user_self_out = torch.cat([pos_user_self_out, neg_user_self_out], dim=1)
            user_aware_out = torch.cat([pos_user_aware_out, neg_user_aware_out], dim=1)
            agent_self_out = torch.cat([pos_agent_self_out, neg_agent_self_out], dim=1)
            agent_aware_out = torch.cat([pos_agent_aware_out, neg_agent_aware_out], dim=1)
            logit_user = []
            logit_agent = []
            for i in range(self.sample_num + 1):
                cos_user = self.calc_cos(user_self_out[:, i, :], user_aware_out[:, i, :])
                cos_agent = self.calc_cos(agent_self_out[:, i, :], agent_aware_out[:, i, :])
                logit_user.append(cos_user)
                logit_agent.append(cos_agent)

            logit_agent = torch.stack(logit_agent, dim=1)
            logit_user = torch.stack(logit_user, dim=1)
        return decoder_outputs, logit_user, logit_agent, contrast_labels
