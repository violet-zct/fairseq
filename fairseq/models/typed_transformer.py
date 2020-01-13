# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from fairseq.modules.typed_transformer_layer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, max_source_positions,
                 encoder_layers, encoder_embed_dim,
                 encoder_attention_heads, encoder_ffn_embed_dim,):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        if embed_tokens is None:
            self.padding_idx = 0
            embed_dim = encoder_embed_dim
        else:
            self.padding_idx = embed_tokens.padding_idx
            embed_dim = embed_tokens.embedding_dim
        self.max_source_positions = max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(encoder_embed_dim,
                 encoder_attention_heads, encoder_ffn_embed_dim, args)
            for i in range(encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.use_seg_pos_emb = getattr(args, 'use_seg_pos_emb', 0)
        if self.use_seg_pos_emb:
            self.seg_pad_idx = 2
            self.seg_pos_emb = Embedding(3, embed_dim, padding_idx=self.seg_pad_idx)

    def forward_embedding(self, src_tokens, seg_pos=-1):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.use_seg_pos_emb:
            masked_src_tokens = src_tokens.masked_fill(src_tokens.ne(self.padding_idx), seg_pos)
            masked_src_tokens = masked_src_tokens.masked_fill(src_tokens.eq(self.padding_idx), self.seg_pad_idx)
            x = x + self.embed_scale * self.seg_pos_emb(masked_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens=None, cls_input=None, return_all_hiddens=False, src_encodings=None,
                encoder_padding_mask=None, attn_mask=None, auxilary_tokens=None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            src_encodings (torch.FloatTensor): shape of `(T x B x C)`
            encoder_padding_mask (torch.Boolean): shape of '(B x T)', where paddings are True
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        if self.embed_tokens is not None:
            x, encoder_embedding = self.forward_embedding(src_tokens, seg_pos=0)
            aug_x, _ = self.forward_embedding(auxilary_tokens, seg_pos=1)
            x = torch.cat([x, aug_x], dim=1)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            src_tokens = torch.cat([src_tokens, auxilary_tokens], dim=1)
            # compute padding mask
            encoder_padding_mask = (src_tokens.eq(self.padding_idx) | src_tokens.eq(self.dictionary.bos_index))
        else:
            assert encoder_padding_mask is not None
            src_tokens = encoder_padding_mask.long()
            encoder_embedding = None
            x = src_encodings
            if self.embed_positions is not None:
                x = x + self.embed_positions(src_tokens).transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, attn_mask=attn_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out.get('encoder_states', None) is not None:
            for idx, state in enumerate(encoder_out['encoder_states']):
                encoder_out['encoder_states'][idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, decoder_embed_dim, decoder_attention_heads,
                 decoder_ffn_embed_dim, decoder_output_dim, max_target_positions, decoder_layers,
                 encoder_embed_dim=-1, no_encoder_attn=False,):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = decoder_embed_dim
        self.output_embed_dim = decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, decoder_embed_dim, decoder_attention_heads,
                                    decoder_ffn_embed_dim, encoder_embed_dim, no_encoder_attn,)
            for _ in range(decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state, **extra_args,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        cur_tgt_pos = None if incremental_state is None else prev_output_tokens.size(1)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if not self_attn_padding_mask.any() and not self.cross_self_attention:
            self_attn_padding_mask = None

        # decoder layers
        attn = None
        attns = []
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out['encoder_states'][idx]
                else:
                    encoder_state = encoder_out['encoder_out']
                if 'global_quantize' in encoder_out:
                    global_vector = encoder_out['global_quantize']  # 1 x B x C
                else:
                    global_vector = None
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn = layer(
                x,
                encoder_state,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=(idx == alignment_layer or not self.training),
                need_head_weights=(idx == alignment_layer),
                cur_tgt_pos=cur_tgt_pos,
                global_vector=global_vector,
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float()
                attns.append(attn.mean(dim=0))

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attns, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m