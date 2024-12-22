import math

import torch
import torch.nn as nn


class FuseLayer2(nn.Module):
    def __init__(self, hidden_size):
        super(FuseLayer2,self).__init__()
        self.mid_gate = nn.Linear(2* hidden_size, hidden_size, bias=False)

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

    def equal_forward(self, hidden_states, attention_mask):
        mul_mask = hidden_states * attention_mask.unsqueeze(-1)
        avg_output = torch.sum(mul_mask, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + self.eps)
        return avg_output


class ContrastSummModel(nn.Module):
    def __init__(self, bart_model, args, tokenizer):
        super(ContrastSummModel, self).__init__()
        self.sample_num = 2 * args.sample_num
        self.role_lambda = args.role_lambda
        self.temperature = args.temperature
        self.max_turn_range = args.max_turn_range

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = AVG(eps=1e-6)
        self.bart_model = bart_model
        self.tokenizer = tokenizer
        self.args = args
        self.hidden_size = args.hidden_size
        self.embed_scale = self.hidden_size ** -0.5

        self.user_fuse_layer = FuseLayer2(768)
        self.agent_fuse_layer = FuseLayer2(768)
        #self.wli=nn.Linear(768,768)

        # Initialize weights and apply final processing

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

        #w = torch.matmul(self.wli(user_self_out)* self.embed_scale, agent_self_out.transpose(-1, -2))

        #w = torch.matmul(user_self_out, agent_self_out.transpose(-1, -2))
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

    def forward(
            self,
            input_ids,
            attention_mask,
            labels,
            role_ids=None,
            utt_ids=None,
            decoder_input_ids=None,
            all_input_ids=None,
            all_attention_mask=None,
            all_role_ids=None,
            all_utt_ids=None,
            contrast_labels=None,

    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """

        model_outputs = self.bart_model(input_ids=input_ids, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids, labels=labels)
        cross_loss = model_outputs.loss

        if contrast_labels is not None and all_input_ids is not None:
            pos_encoder_hidden_states = model_outputs.encoder_last_hidden_state
            neg_encoder_outputs = self.bart_model.model.encoder(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask,
                return_dict=True,
            )
            neg_encoder_hidden_states = neg_encoder_outputs.last_hidden_state
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
            loss_agent = self.calc_loss(logit_agent, contrast_labels)
            loss_user = self.calc_loss(logit_user, contrast_labels)
            role_loss = loss_agent + loss_user
            #loss = cross_loss + self.role_lambda * role_loss
            loss = self.role_lambda * cross_loss + (1 - self.role_lambda) * role_loss
        else:
            loss = cross_loss
        return loss

    def generate(self, input_ids, decoder_input_ids, attention_mask):
        gen_kwargs = {
            "max_length": self.args.max_tgt_len,
            #"min_length": self.args.min_tgt_len,
            "early_stopping": True,
            "num_beams": self.args.num_beams,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        return self.bart_model.generate(input_ids=input_ids, decoder_input_ids=decoder_input_ids,
                                        attention_mask=attention_mask, **gen_kwargs)
