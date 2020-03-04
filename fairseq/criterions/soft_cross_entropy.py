# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch


def label_smoothed_nll_loss(lprobs, target, epsilon, non_pad_mask=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if non_pad_mask is not None:
        #non_pad_mask = target.ne(ignore_index)
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


@register_criterion('soft_cross_entropy')
class SoftCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eos_idx = args.codebook_size
        self.eps = args.label_smoothing if args.label_smoothing > 0 else 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        shifted_src_tokens, lengths, target_tokens = sample['net_input']['src_tokens'], \
                                                     sample['net_input']['src_lengths'], sample['target']
        self.task.vqvae_model.eval()
        with torch.no_grad():
            # codes are weighted one_hot matrix: B x T x |V|
            # mask: B x T, set pad to be False
            codes, mask = self.task.vqvae_model.forward_encoder(target_tokens, lengths, extract_code_only=True,
                                                              code_extract_strategy=getattr(self.args,
                                                                                            'code_extract_strategy',
                                                                                            None))
            codes = torch.argmax(codes, dim=-1)
            batch = mask.size(0)
            code_num = mask.sum(1).eq(mask.size(1))

            if torch.any(code_num):
                # 1 3 x      1 3 x e
                # 2 x x  ->  2 x x e
                # 3 4 5      3 4 5 e
                mask = torch.cat([mask, mask.new_full((batch, 1), False)], dim=-1)
                codes = torch.cat([codes, codes.new_full((batch, 1), self.eos_idx)], dim=-1)

            # before appending eos to each sentence, we create src_tokens by right-shift 1
            # 1 3 x e      e 1 3 x
            # 2 x x e  ->  e 2 x x
            # 3 4 5 e      e 3 4 5
            src_tokens = torch.cat([codes.new_full((codes.size(0), 1), self.eos_idx), codes[:, :-1]], dim=1)
            # 1 3 x e      1 3 e e
            # 2 x x e  ->  2 e x e
            # 3 4 5 e      3 4 5 e
            sort_index = torch.argsort(mask.float(), -1, descending=False)[:, 0]
            assert not torch.any(sort_index.eq(0)), 'the first word is eos!'
            # set the last index to be eos
            codes[torch.arange(batch).to(codes), sort_index] = self.eos_idx
            mask[torch.arange(batch).to(codes), sort_index] = True  # set actual tokens to be True
            # eos symbol is the last index added to the code book
            prev_output_masks = ~mask  # mask sets padding to be True
            target = codes

        logits, _ = model(src_tokens, prev_output_masks=prev_output_masks)
        lprobs = utils.log_softmax(logits, dim=-1)

        if self.eps > 0:
            loss, nll_loss = self.compute_label_smooth_loss(lprobs, target, mask)
        else:
            nll_loss = loss = self.compute_loss(lprobs, target, mask)

        sample_size = mask.sum().item()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample_size,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_label_smooth_loss(self, lprobs, target, non_pad_mask=None, reduce=True):
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.view(-1, 1), self.eps, non_pad_mask=non_pad_mask.view(-1, 1), reduce=reduce,
        )
        return loss, nll_loss

    def compute_loss(self, lprobs, target, non_pad_mask=None, reduce=True):
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        non_pad_mask = non_pad_mask.view(-1, 1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        if non_pad_mask is not None:
            nll_loss = nll_loss[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
        return nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
