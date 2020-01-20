# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch
import torch.nn.functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('vqvae_label_smoothed_cross_entropy')
class VQVAELabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.commitment_cost = args.commitment_cost
        self.commit_warm_up_steps = args.commitment_cost_warm_up_steps
        self.commit_anneal_steps = args.commitment_cost_anneal_steps
        self.updates = 0
        self.latent_k = args.bottom_latent_k
        self.word_predict_loss_weight = getattr(args, 'aug_loss_weight', 0)

        self.at_prior_loss_start_steps = getattr(args, 'at_prior_loss_start_steps', 0)
        self.at_prior_loss_anneal_steps = getattr(args, 'at_prior_loss_anneal_steps', 0)
        self.at_prior_loss_min_weight = getattr(args, 'at_prior_loss_min_weight', 0)
        self.at_prior_loss_max_weight = getattr(args, 'at_prior_loss_max_weight', 0)
        self.world_size = args.distributed_world_size

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--commitment-cost', default=0.25, type=float,
                            help="weight of commitment cost between E(x) and sg(e)")
        parser.add_argument('--commitment-cost-warm-up-steps', default=0, type=float)
        parser.add_argument('--commitment-cost-anneal-steps', default=0, type=float)
        # fmt: on

    def get_commitment_weight(self):
        if self.updates <= self.commit_warm_up_steps:
            return self.commitment_cost
        elif self.commit_anneal_steps == 0:
            return self.commitment_cost
        else:
            return self.commitment_cost * min((self.updates - self.commit_warm_up_steps) / self.commit_anneal_steps, 1.0)

    def get_at_prior_loss_weight(self):
        if self.updates <= self.at_prior_loss_start_steps:
            return 0.
        else:
            increase_steps = min((self.updates - self.at_prior_loss_start_steps), self.at_prior_loss_anneal_steps) * 1.0
            return min(self.at_prior_loss_max_weight, self.at_prior_loss_min_weight + increase_steps/self.at_prior_loss_anneal_steps)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # what is the sample size is different for different losses?

        shifted_src_tokens, lengths, target_tokens = sample['net_input']['src_tokens'], \
                                                     sample['net_input']['src_lengths'], sample['target']
        #  logits, diff, quantize_stats, mask.sum().type_as(diff), codes, quantize_out['word_predict']
        net_output = model(shifted_src_tokens, lengths, target_tokens, self.updates)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)  # loss is the sum loss over tokens
        commit_weight = self.get_commitment_weight()

        commitment_loss = commit_weight * net_output[1]  # this is the mean loss over latent dimensions
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        commitment_loss = commitment_loss * sample_size * self.world_size
        loss = loss + commitment_loss

        word_predict = net_output[5]
        if word_predict is not None:
            word_predict_loss = self.compute_cross_entropy(word_predict, sample, reduce=reduce)  # loss is the sum loss over tokens
            loss += self.word_predict_loss_weight * word_predict_loss

        code_prior = net_output[6]
        code_prior_logits, code_prior_gold = code_prior['code_prior_logits'], code_prior['code_prior_gold']
        if code_prior_logits is not None and code_prior_gold is not None:
            code_prior_loss, code_prior_nll_loss = self.compute_label_smooth_loss(code_prior_logits, code_prior_gold, model.bottom_quantizer.n_embed)
            actual_codes = net_output[3]
            batch_size = sample['target'].size(0)
            code_prior_loss = code_prior_loss / (actual_codes - batch_size) * sample_size * self.world_size
            code_prior_nll_loss = code_prior_nll_loss / (actual_codes - batch_size) * sample_size * self.world_size
            loss += self.get_at_prior_loss_weight() * code_prior_loss

        # nll_loss is sum over all the tokens/sentences
        true_nll_loss = nll_loss + math.log(self.latent_k) * net_output[3]  # net_output[3] is the actual number of codes

        quantize_stats = {k: utils.item(v.data) if not isinstance(v, int) else v for k, v in net_output[2].items()}
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'commit_loss': utils.item(commitment_loss.data),
            'true_nll_loss': utils.item(true_nll_loss.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'code_num': utils.item(net_output[3].data),
        }

        if word_predict is not None:
            logging_output['word_nll_loss'] = utils.item(word_predict_loss.data) if reduce else word_predict_loss.data

        if code_prior_logits is not None and code_prior_gold is not None:
            logging_output['code_prior_loss'] = utils.item(code_prior_loss.data) if reduce else code_prior_loss.data
            logging_output['code_prior_nll_loss'] = utils.item(code_prior_nll_loss.data) if reduce else code_prior_nll_loss.data

        if not self.training:
            codes = net_output[4]['bottom_codes']  # B x T
            # logging_output['unique_codes'] = torch.unique(codes)
            logging_output['unique_codes'] = len(torch.unique(codes))

        logging_output.update(quantize_stats)
        if self.training:
            self.updates += 1
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_xet_loss(self, logits, gold, padding_idx, reduce=True):
        lprobs = utils.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = gold.contiguous().view(-1, 1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    def compute_cross_entropy(self, logits, sample, reduce=True):
        lprobs = utils.log_softmax(logits, dim=-1, onnx_trace=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        code_num = sum(log.get('code_num', 0) for log in logging_outputs)

        results = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'commit_loss': sum(log.get('commit_loss', 0) for log in logging_outputs) / len(logging_outputs) / sample_size if sample_size > 0 else 0.,
            'true_nll_loss': sum(log.get('true_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'code_num': code_num / len(logging_outputs),
        }

        if len(logging_outputs) > 0 and 'word_nll_loss' in logging_outputs[0]:
            results['word_nll_loss'] = sum(log.get('word_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.

        if len(logging_outputs) > 0 and 'unique_codes' in logging_outputs[0]:
            codes = sum(log.get('unique_codes', 0) for log in logging_outputs) / len(logging_outputs)
            results['unique_codes'] = codes

        if len(logging_outputs) > 0 and 'code_prior_loss' in logging_outputs[0]:
            results['code_prior_loss'] = sum(log.get('code_prior_loss', 0) for log in logging_outputs) / len(logging_outputs) \
                                         / sample_size if sample_size > 0 else 0.
            results['code_prior_nll_loss'] = sum(log.get('code_prior_nll_loss', 0) for log in logging_outputs) / len(logging_outputs) \
                                             / ntokens / math.log(2) if ntokens > 0 else 0.

        if len(logging_outputs) > 0:
            for k in logging_outputs[0].keys():
                if k not in results:
                    results[k] = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)

        return results