# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch


@register_criterion('soft_cross_entropy')
class SoftCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        shifted_src_tokens, lengths, target_tokens = sample['net_input']['src_tokens'], \
                                                     sample['net_input']['src_lengths'], sample['target']
        with torch.no_grad():
            # codes are weighted one_hot matrix: B x T x |V|
            # mask: B x T, set pad to be False
            codes, mask = self.task.vqvae_model.forward_encoder(target_tokens, lengths, extract_code_only=True,
                                                              code_extract_strategy=getattr(self.args,
                                                                                            'code_extract_strategy',
                                                                                            None))
        if codes.size(1) == 0:
            raise ValueError
        else:
            if codes.size(1) == 1:
                src_tokens = codes
                prev_output_masks = ~mask  # mask sets padding to be True
                target, target_mask = codes, mask
            else:
                src_tokens = codes[:, :-1]
                prev_output_masks = ~mask[:, :-1]
                target, target_mask = codes[:, 1:], mask[:, 1:]
            logits, _ = model(prev_output_tokens=src_tokens, prev_output_masks=prev_output_masks)
            lprobs = utils.log_softmax(logits, dim=-1)
            loss = (target * lprobs).sum(-1) * (target_mask.type_as(lprobs))
            loss = -loss.sum()
            sample_size = target_mask.sum().item()
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample_size,
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
        return loss, sample_size, logging_output

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
