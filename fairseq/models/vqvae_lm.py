from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.typed_transformer import (
    Embedding,
    TransformerEncoder,
)
from fairseq.models.conv_encoder import (
    ConvEncoder,
    FullConvEncoder,
    SingleKernelFullConvEncoder,
    SingleKernelFullDeConvEncoder,
)
from fairseq.modules import PositionalEmbedding

import torch
from torch.nn import functional as F
from torch import nn

import numpy as np
import math
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def parse_kernel_and_strides(kernel, stride):
    def _parse(inpt):
        return list(map(int, inpt.split(",")))

    def _parse_kernels(inpt):
        return [list(map(int, ii.split('-'))) for ii in inpt.split(",")]

    kernels = _parse_kernels(kernel)
    check = [len(ii) == 1 for ii in kernels]
    if all(check):
        kernels = [ii[0] for ii in kernels]
    return kernels, _parse(stride)


def print_stats(stats):
    for k, v in stats.items():
        print("{} = {}".format(k, v.item()))


class Quantize(nn.Module):
    def __init__(self, args, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.soft = args.soft_em
        self.max_temp = args.soft_max_temp
        self.min_temp = args.soft_min_temp
        self.diff_temp = self.max_temp - self.min_temp
        self.anneal_steps = args.soft_temp_anneal_steps
        self.samples = args.soft_samples

        # embed = torch.randn(dim, n_embed)
        embed = torch.normal(mean=0, std=dim ** -0.5, size=(dim, n_embed))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

        self.exploration_steps = args.quantize_explore_steps
        self.most_attend_k = int(self.n_embed * 0.6)

    def get_temperature(self, updates):
        if updates == -1:
            return self.min_temp
        return self.min_temp if updates >= self.anneal_steps else ((updates * 1.0 / self.anneal_steps) * self.diff_temp + self.min_temp)

    def forward(self, input, input_mask, updates=-1, prefix=""):
        '''
        :param input: T x batch x C, number of channels: dimension C
        :param input_mask: T x batch
        :return:
        '''
        # S = T x B
        flatten = input.reshape(-1, self.dim)  # S x C
        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # S x 1
            - 2 * flatten @ self.embed   # S x C @ C x K
            + self.embed.pow(2).sum(0, keepdim=True)  # 1 x K
        )  # S x K

        if self.training and updates < self.exploration_steps:
            sorted_cluster_size, sorted_indices = torch.sort(self.cluster_size, descending=True)
            most_attend_codes = sorted_indices[:self.most_attend_k]
            dist = dist.index_fill_(1, most_attend_codes, float('inf'))

        if self.soft and self.training:
            tau = self.get_temperature(updates)
            embed_ind = torch.multinomial(F.softmax(-dist / tau, -1), self.samples, replacement=True)  # S x samples
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type_as(flatten).mean(1)  # S x samples x K -> S x K
            quantize = (embed_onehot @ self.embed.transpose(0, 1)).view(input.size(0), input.size(1), self.dim)
        else:
            _, embed_ind = (-dist).max(1)  # S
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type_as(flatten)  # S x K
            quantize = self.embed_code(embed_ind.view(*input.shape[:-1]))  # T X batch x C

        # todo: this is for debugging, comment it later
        stats = {}
        # batch = embed_ind.size(1)
        # lengths = torch.sum(input_mask, dim=0)
        # avg = input.new_zeros((batch))
        # masked_embed_inds = embed_ind * input_mask.type_as(embed_ind) + torch.ones_like(embed_ind) * -1 * (1 - input_mask.type_as(embed_ind))
        # for ii in range(batch):
        #     avg[ii] = len(torch.unique(masked_embed_inds[:, ii])) - 1
        # tot_unique_per_batch = len(torch.unique(masked_embed_inds)) - 1
        # avg_unique_per_example = torch.sum(avg / lengths.type_as(avg)) / batch
        # stats['tot unique latents per batch'] = tot_unique_per_batch
        # stats['avg unique latents per example'] = avg_unique_per_example

        effective_units = 1.0 / embed_onehot[input_mask.view(-1)].mean(0).pow(2).sum()
        stats[prefix + 'effective latents per batch'] = effective_units
        if self.training:
            unmasked_flatten = torch.masked_select(flatten, input_mask.view(-1, 1)).contiguous().view(-1, self.dim)  # num_latents x C
            unmasked_embed_onehot = torch.masked_select(embed_onehot, input_mask.view(-1, 1)).contiguous().view(-1, self.n_embed)  # num_latents x K

            cluster_sum = unmasked_embed_onehot.sum(0)
            embed_sum = unmasked_flatten.transpose(0, 1) @ unmasked_embed_onehot  # C x K

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce(cluster_sum)
                torch.distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, cluster_sum
            )
            self.embed_avg.data.mul_(self.decay).add_(
                1 - self.decay, embed_sum
            )
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            # stats[prefix + 'moving_avg_cluster_mean'] = torch.mean(self.cluster_size)
            # stats[prefix + 'moving_avg_cluster_var'] = torch.var(self.cluster_size)

        input_mask = input_mask.type_as(input)
        quantize = quantize * input_mask.unsqueeze(-1)
        input = input * input_mask.unsqueeze(-1)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind, stats

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


@register_model('vqvae_lm')
class VQVAE(FairseqLanguageModel):
    def __init__(self, args, text_encoder, text_conv_encoder, text_decoder, bottom_quantizer, bottom_latent_encoder, global_quantizer):
        super().__init__(text_decoder)
        self.text_encoder = text_encoder
        self.text_conv_encoder = text_conv_encoder
        self.bottom_quantizer = bottom_quantizer
        self.bottom_latent_encoder = bottom_latent_encoder
        self.global_quantizer = global_quantizer

        self.bottom_conv_kernel_size, self.bottom_conv_strides = \
            parse_kernel_and_strides(args.bottom_conv_kernel_size, args.bottom_conv_stride)

        self.pad_index = text_decoder.padding_idx

        self.add_latent_position = args.add_latent_positions
        if self.add_latent_position:
            self.latent_positional_embedding = PositionalEmbedding(
            args.max_source_positions, self.bottom_quantizer.dim, self.pad_index,
            learned=args.decoder_learned_pos,
        )

        self.word_drop_rate = args.drop_word_prob
        self.pretrain_steps = args.pretrain_steps
        self.encoder_context_window = args.context_window

        self.encoder_opt_dropout = getattr(args, 'encoder_opt_dropout', 0.0)
        self.encoder_form = getattr(args, 'encoder_form', 'mix')
        self.shrink_ratio = getattr(args, 'shrink_ratio', 8)

        self.args = args
        # todo: add append
        if args.use_deconv:
            self.aug_input = 'add'
        else:
            self.aug_input = 'none'  # prepend as prefix with mask?

        self.aug_loss = getattr(args, 'aug_loss', None)
        if self.aug_loss is not None:
            self.share_emb = getattr(args, 'share_aug_softmax_with_emb', 0)
            if self.aug_loss == 'code_bow':
                self.latent_positional_embedding = PositionalEmbedding(
                    self.shrink_ratio, self.bottom_quantizer.dim, self.shrink_ratio+1,
                    learned=args.decoder_learned_pos,)
            if not self.share_emb:
                self.word_predict_out = nn.Parameter(torch.Tensor(len(self.decoder.dictionary), args.encoder_embed_dim))
                nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off

        ### general arguments for all components
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')

        # TODO: add pretrained latent embedding path
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')

        # arguments for text encoder (transformer)
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # arguments for bottom level convolutional text encoder
        parser.add_argument('--bottom-conv-kernel-size', type=str,
                            help='[format: 1,2,3], kernel sizes for the bottom latent variables conv layers, '
                                 'with transformer encoder outputs as inputs,'
                                 'to use multi-channel CNN, use the format of [2-3-4,5-7]')
        parser.add_argument("--bottom-conv-stride", type=str,
                            help='[format: 2,2,2], strides for the bottom latent variables conv layers')

        # arguments for the bottom level quantization
        parser.add_argument('--bottom-latent-dim', type=int,
                            help='bottom code book dimension')
        parser.add_argument('--bottom-latent-k', type=int,
                            help='bottom code book size')

        # shared arguments for code books (soft em)
        parser.add_argument('--soft-em', type=int,
                            help='use soft EM for bottom latent codes')
        parser.add_argument('--soft-max-temp', type=float,
                            help='0 < tau < 1, makes the softmax sharper; tau > 1, makes the softmax smoother')
        parser.add_argument('--soft-min-temp', type=float,
                            help='0 < tau < 1, makes the softmax sharper; tau > 1, makes the softmax smoother')
        parser.add_argument('--soft-temp-anneal-steps', type=int,
                            help='0, no anneal, take min temp; otherwise anneal from max to min temp within this steps')
        parser.add_argument('--soft-samples', type=int,
                            help='number of samples used in multinomial distribution to create soft samples')

        # todo: condition (attention) on the top level discrete representations and add condition to the typed TransformerEncoderLayer
        # arguments for the bottom level discrete latent variable encoder (transformer)
        #--bottom-encoder-embed-dim = --bottom-latent-dim
        parser.add_argument('--use-bottom-quantants-encoder', type=int, metavar='N')
        parser.add_argument('--bottom-encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--bottom-encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--bottom-encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # arguments for text decoder (transformer)
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')

        parser.add_argument('--drop-word-prob', type=float,
                            help='probability of drop words of decoder inputs')  # todo: remove

        # other discrete variables
        parser.add_argument('--use-global-quantant', type=int,
                            help='if use bos to create global quantization vector')
        parser.add_argument('--global-latent-dim', type=int,
                            help='global representaion code book dimension')
        parser.add_argument('--global-latent-k', type=int,
                            help='global code book size')
        # ablations
        parser.add_argument('--pretrain-steps', type=int, metavar='N')  # todo: remove
        parser.add_argument('--use-latent', type=int, metavar='N')
        parser.add_argument('--add-latent-positions', type=int)
        parser.add_argument('--context-window', type=int, default=0)
        parser.add_argument('--encoder-opt-dropout', type=float, default=0.)
        parser.add_argument('--encoder-form', type=str, default='mix', choices=['mix', 'conv', 'append'],
                            help='mix=transformer+cnn, conv=fully cnn, append=transformer+pending positional embedding')
        parser.add_argument('--shrink-ratio', type=int, default=8,
                            help='used for append')

        parser.add_argument('--use-deconv', type=int, default=0,
                            help='apply deconvolution to latent vectors')
        parser.add_argument('--quantize-explore-steps', type=int, default=-1,)
        # this can learn semantic mapping?
        parser.add_argument('--aug-loss', default=None, choices=['deconv_predict', 'code_bow'])
        parser.add_argument('--aug-loss-weight', default=0.2, type=float)
        parser.add_argument('--share-aug-softmax-with-emb', default=0, type=int)
        parser.add_argument('--use-seg-pos-emb', default=0, type=int,
                            help='used when encoder form is append')
        parser.add_argument('--use-stride-first', default=1, type=int)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
            args.max_target_positions = DEFAULT_MAX_SOURCE_POSITIONS
        else:
            args.max_source_positions = args.max_source_positions
            args.max_target_positions = args.max_source_positions

        src_dict = task.source_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # though there is only one language for the LM, we still stick to encoder and decoder for clarity
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                src_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        text_encoder, text_conv_encoder, bottom_latent_encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        text_decoder = cls.build_decoder(args, src_dict, decoder_embed_tokens)

        if args.use_latent:
            bottom_quantizer = cls.build_quantizer(args)
        else:
            bottom_quantizer = None

        if args.use_global_quantant:
            if args.global_latent_dim != args.decoder_embed_dim:
                assert args.decoder_embed_dim % args.global_latent_dim == 0
                k_global_codes = args.decoder_embed_dim // args.global_latent_dim
            else:
                k_global_codes = 1
            global_quantizer = nn.ModuleList([Quantize(args, args.global_latent_dim, args.global_latent_k) for _ in range(k_global_codes)])
        else:
            global_quantizer = None
        return VQVAE(args, text_encoder, text_conv_encoder, text_decoder, bottom_quantizer, bottom_latent_encoder,
                     global_quantizer)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if args.encoder_form != 'append':
            kernels, strides = parse_kernel_and_strides(args.bottom_conv_kernel_size, args.bottom_conv_stride)
        args.shrink_ratio = np.prod(strides)
        if args.encoder_form == 'mix':
            text_encoder = TransformerEncoder(args, src_dict, embed_tokens, args.max_source_positions,
                                              args.encoder_layers, args.encoder_embed_dim, args.encoder_attention_heads,
                                              args.encoder_ffn_embed_dim)
            text_conv_encoder = ConvEncoder(args.encoder_embed_dim, kernels, strides, args.bottom_latent_dim)
        elif args.encoder_form == 'conv':
            if isinstance(kernels[0], int):
                # kernels are the same
                text_encoder = SingleKernelFullConvEncoder(args, args.encoder_embed_dim, kernels, strides, args.bottom_latent_dim,
                                           embed_tokens, src_dict)
            else:
                # multiple-kernels in one layer
                text_encoder = FullConvEncoder(args, args.encoder_embed_dim, kernels, strides, args.bottom_latent_dim,
                                           embed_tokens, src_dict)
            if args.use_deconv:
                # let's misuse the conv_encoder for deconvolutional encodder
                text_conv_encoder = SingleKernelFullDeConvEncoder(args.encoder_embed_dim, kernels[::-1], strides[::-1], src_dict)
            else:
                text_conv_encoder = None
        elif args.encoder_form == 'append':
            text_encoder = TransformerEncoder(args, src_dict, embed_tokens, args.max_source_positions,
                                              args.encoder_layers, args.encoder_embed_dim, args.encoder_attention_heads,
                                              args.encoder_ffn_embed_dim)
            text_conv_encoder = None
        else:
            raise NotImplementedError

        if args.use_bottom_quantants_encoder:
            bottom_latent_encoder = TransformerEncoder(args, None, None, args.max_source_positions,
                                            args.bottom_encoder_layers, args.bottom_latent_dim,
                                            args.bottom_encoder_attention_heads, args.bottom_encoder_ffn_embed_dim)
        else:
            bottom_latent_encoder = None
        return text_encoder, text_conv_encoder, bottom_latent_encoder

    @classmethod
    def build_quantizer(cls, args):
        bottom_quantizer = Quantize(args, args.bottom_latent_dim, args.bottom_latent_k)
        return bottom_quantizer

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.no_cross_attention:
            from fairseq.models.transformer import TransformerDecoder
            text_decoder = TransformerDecoder(
                args, tgt_dict, embed_tokens, no_encoder_attn=True,
            )
        else:
            from fairseq.models.typed_transformer import TransformerDecoder
            text_decoder = TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                args.decoder_embed_dim, args.decoder_attention_heads,
                args.decoder_ffn_embed_dim, args.decoder_output_dim,
                args.max_source_positions, args.decoder_layers,
                encoder_embed_dim=args.bottom_latent_dim,
                no_encoder_attn=getattr(args, 'no_cross_attention', False),
            )
        return text_decoder

    def mask_words(self, src_tokens, lengths):
        batch = src_tokens.size(0)
        src_masks = src_tokens.eq(self.pad_index) | src_tokens.eq(self.decoder.dictionary.bos())
        full_length = src_tokens.size(1)
        if full_length <= 2:
            return src_tokens
        mask_lengths = (lengths.float() * self.word_drop_rate).long()
        mask = torch.arange(full_length).to(src_tokens.device).unsqueeze(0).expand(batch, full_length).ge(
            mask_lengths.unsqueeze(1))
        mask = mask.long()
        scores = src_tokens.clone().float().uniform_()
        scores.masked_fill_(src_masks, -1)
        sorted_values, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        mask = mask.scatter(1, sorted_idx, mask)  # 0 are dropped words
        src_tokens[(1 - mask).bool()] = self.pad_index
        return src_tokens

    def spacing_mask_words(self, src_tokens, lengths):
        if np.random.uniform() > 0.85:
            return src_tokens
        valid_lengths = (lengths - 1) / self.shrink_ratio + 1
        max_length = torch.max(valid_lengths).item()
        max_src_length = src_tokens.size(1)
        valid_index = torch.arange(max_length, device=lengths.device).type_as(lengths).expand(len(lengths), max_length)
        valid_start_idx = valid_index * self.shrink_ratio
        rand1 = torch.randint(1, min(self.shrink_ratio, max_src_length), (src_tokens.size(0),))
        rand = torch.randint_like(valid_start_idx, 0, self.shrink_ratio)
        rand[:, 0] = rand1
        source_mask_index = rand + valid_start_idx
        unmask_local_mask = ((source_mask_index >= max_src_length) | (source_mask_index <= 0))
        first_index = source_mask_index[:, 0].unsqueeze(1).expand(len(lengths), max_length)
        source_mask_index = source_mask_index * ((~unmask_local_mask).long()) + first_index * unmask_local_mask.long()
        assert torch.all(source_mask_index < max_src_length)
        src_tokens = src_tokens.scatter(1, source_mask_index, self.pad_index)
        return src_tokens

    def create_mask(self, tokens):
        if self.encoder_context_window == 0:
            return None
        else:
            T = tokens.size(1)
            a = torch.arange(T).type_as(tokens).unsqueeze(0).expand(T, T)
            b = (torch.arange(T).type_as(tokens) - self.encoder_context_window).unsqueeze(1).expand(T, T)
            c = (torch.arange(T).type_as(tokens) + self.encoder_context_window).unsqueeze(1).expand(T, T)
            return (~((a <= c) | (a >= b))).float()

    def create_aug_input(self, src_tokens, lengths):
        append_lengths = (lengths - 1) / self.shrink_ratio + 1
        max_length = torch.max(append_lengths).item()
        mask = torch.arange(max_length, device=lengths.device).type_as(lengths).expand(len(lengths), max_length)
        mask = mask < append_lengths.unsqueeze(1)  # B x L
        aug_tokens = src_tokens.new_ones((src_tokens.size(0), max_length)).fill_(self.text_encoder.dictionary.bos_index)
        aug_tokens = aug_tokens.masked_fill(~mask, self.text_encoder.dictionary.pad_index)
        return aug_tokens, mask

    def forward(self, decoder_tokens, lengths, full_tokens, update_steps, **kwargs):
        mask, diff, quantize_out, quantize_stats, codes = self.forward_encoder(full_tokens, lengths, update_steps)
        if self.training and self.word_drop_rate > 0.0:
            decoder_tokens = self.spacing_mask_words(decoder_tokens, lengths)
            # decoder_tokens = self.mask_words(decoder_tokens, lengths)
        decoder_out = self.forward_decoder(decoder_tokens, encoder_out=quantize_out)
        logits = decoder_out[0]
        return logits, diff, quantize_stats, mask.sum().type_as(diff), codes, quantize_out['word_predict']

        # todo: prior model - one decoder, one encoder-decoder
        # todo: hierarchical model

    def forward_encoder(self, full_tokens, lengths, update_steps=-1, extrac_code_only=False):
        """
                output of text encoder
                {
                    'encoder_out': x,  # T x B x C
                    'encoder_padding_mask': encoder_padding_mask,  # B x T
                    'encoder_embedding': encoder_embedding,  # B x T x C
                    'encoder_states': encoder_states,  # List[T x B x C]
                }
                """
        pad_num = 0
        max_len = full_tokens.size(1)

        if self.encoder_form == 'mix':
            encoder_attn_mask = self.create_mask(full_tokens)
            text_encoder_out = self.text_encoder(full_tokens, attn_mask=encoder_attn_mask)
            encodings = text_encoder_out['encoder_out']

            # encoding_mask = (~text_encoder_out['encoder_padding_mask']).type_as(text_encoder_out['encoder_out'])
            encoding_mask = (~(full_tokens.eq(self.pad_index) | full_tokens.eq(self.decoder.dictionary.bos()))).type_as(
                text_encoder_out['encoder_out'])
            conv_inpt = encodings * (encoding_mask.transpose(0, 1).unsqueeze(-1))  # T x B x C

            # !!!!!!!!!!!!! the output mask sets padding to be False
            # text_conv_out: T' x batch x C'
            # mask: batch x T'
            text_conv_out, mask = self.text_conv_encoder(conv_inpt.permute(1, 2, 0),
                                                         lengths)  # B x C x T -> T' x B x C', C' = latent_dim
        elif self.encoder_form == 'conv':
            if self.args.use_deconv:
                if (max_len - 1) % self.shrink_ratio != 0:
                    pad_num = math.ceil((max_len - 1) / self.shrink_ratio) * self.shrink_ratio + 1 - max_len
                    full_tokens = torch.cat([full_tokens, full_tokens.new_full((full_tokens.size(0), pad_num), self.pad_index)], dim=1)
            text_conv_out, mask = self.text_encoder(full_tokens, lengths, pad_num)
        elif self.encoder_form == 'append':
            aug_tokens, mask = self.create_aug_input(full_tokens, lengths)
            text_encoder_out = self.text_encoder(full_tokens, auxilary_tokens=aug_tokens)
            text_conv_out = text_encoder_out['encoder_out'][full_tokens.size(1):, :, :]  # T' x B x C
        else:
            raise NotImplementedError

        if self.add_latent_position:
            dummy_batch = full_tokens.new_zeros((mask.size(0), mask.size(1))).masked_fill(~mask, self.pad_index)
            text_conv_out = text_conv_out + self.latent_positional_embedding(dummy_batch).transpose(0, 1)

        if self.encoder_opt_dropout > 0.0:
            text_conv_out = F.dropout(text_conv_out, p=self.encoder_opt_dropout, training=self.training)

        word_predict = None
        if self.bottom_quantizer is not None and (update_steps > self.pretrain_steps or not self.training):
            # diff is the loss to update the enocder
            # quantize: masked T X batch x C; diff: scalar; embed_ind: T x batch
            quantize, diff, embed_ind, quantize_stats = self.bottom_quantizer(text_conv_out,
                                                                              mask.transpose(0, 1).contiguous(),
                                                                              updates=update_steps)
            # assume currently we are using deconv, so the input are padded to be multiple times of the shrink_ratio + 1
            if self.training and self.aug_loss == 'code_bow':
                # T' x batch x C -> batch x T x C
                T_prime, batch_size, D = quantize.size()
                expand_quantize = quantize.transpose(0, 1).unsqueeze(2).expand(batch_size, T_prime, self.shrink_ratio, D).reshape(batch_size, -1, D)
                expand_quantize = expand_quantize[:, :max_len, :]  # B x T x C
                code_index = torch.arange(max_len).unsqueeze(0).type_as(full_tokens) % self.shrink_ratio
                pos_emb = self.latent_positional_embedding(code_index)  # 1 x T x C
                expand_quantize = pos_emb + expand_quantize
                if self.share_emb:
                    word_predict = F.linear(expand_quantize, self.decoder.embed_tokens.weight)
                else:
                    word_predict = F.linear(expand_quantize, self.word_predict_out)

            if self.args.use_deconv and not extrac_code_only:
                deconv_output = self.text_conv_encoder(quantize.permute(1, 2, 0), lengths, pad_num,
                                                       full_tokens.ne(self.pad_index))
                quantize = deconv_output.transpose(1, 2)[:, :max_len, :]  # B x T x C

                if self.training and self.aug_loss == 'deconv_predict':
                    if self.share_emb:
                        word_predict = F.linear(quantize, self.decoder.embed_tokens.weight)
                    else:
                        word_predict = F.linear(quantize, self.word_predict_out)
        else:
            quantize = text_conv_out
            diff = text_conv_out.new_zeros(1)
            quantize_stats = {}

        if self.bottom_latent_encoder is not None:
            quantize = self.bottom_latent_encoder(src_encodings=quantize, encoder_padding_mask=~mask)
            quantize = quantize['encoder_out']

        quantize_out = {'encoder_out': quantize,  # masked T X batch x C  or B x T x C when use deconv
                        'encoder_padding_mask': ~mask,  # B x T, this mask sets padding to be True
                        # 'encoder_embedding': text_encoder_out['encoder_embedding']  # B x T x C
                        'encoder_embedding': None,  # B x T x C
                        'word_predict': word_predict
                        }

        if self.global_quantizer is not None:
            dummy_mask = mask.new_ones((1, text_conv_out.size(1)))  # 1 x B
            gvec = text_conv_out.mean(0)  # B x C
            gquants = []
            gids = []
            for ii, gquant in enumerate(self.global_quantizer):
                d = gquant.dim
                global_quantize, global_diff, global_embed_ind, \
                global_quantize_stats = gquant(gvec[:, ii*d:(ii+1)*d].unsqueeze(0),
                                                dummy_mask,
                                                prefix="global_{}_".format(ii))
                diff = diff + global_diff
                quantize_stats.update(global_quantize_stats)
                gquants.append(global_quantize)

                if gquant.soft:
                    global_embed_ind = F.one_hot(global_embed_ind, gquant.n_embed).sum(1).max(1)
                gids.append(global_embed_ind)  # batch_size
            # quantize = torch.cat([global_quantize, quantize], dim=0)
            # mask = torch.cat([dummy_mask.transpose(0, 1), mask], dim=1)
            quantize_out['global_quantize'] = torch.cat(gquants, dim=-1)

        if not self.training:
            embed_ind = embed_ind.view(*text_conv_out.shape[:-1])  # T x batch
            codes = {'bottom_codes': embed_ind.transpose(0, 1).masked_fill(~mask, -1)}
            if self.global_quantizer is not None:
                codes['global_codes'] = gids
        else:
            codes = None
        return mask, diff, quantize_out, quantize_stats, codes

    def forward_decoder(self, decoder_tokens, encoder_out, incremental_state=None):
        decoder_out = self.decoder(decoder_tokens, encoder_out=encoder_out, incremental_state=incremental_state,
                                   aug_input=self.aug_input)
        return decoder_out

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
            idx = 0 if self.args.use_deconv else 1
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(idx, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out.get('encoder_states', None) is not None:
            for idx, state in enumerate(encoder_out['encoder_states']):
                encoder_out['encoder_states'][idx] = state.index_select(1, new_order)
        return encoder_out

    def quantization(self, codes, code_mask, global_codes=None, extract_codes_only=False):
        # codes: batch x T; code_mask: batch x T; mask here sets pad to be True
        # global_codes: batch
        quantize = self.bottom_quantizer.embed_code(codes)  # batch x T x dim

        # todo: global codes needs to be reimplemented
        if global_codes is not None:
            global_quantize = self.global_quantizer.embed_code(global_codes.unsqueeze(0))  # 1 x batch x dim
            dummy_mask = code_mask.new_zeros((code_mask.size(0)))  # B x 1
            quantize = torch.cat([global_quantize, quantize], dim=0)
            code_mask = torch.cat([dummy_mask.transpose(0, 1), code_mask], dim=1)

        if self.args.use_deconv and not extract_codes_only:
            valid = ~code_mask
            lengths = (torch.sum(valid, dim=1).long() - 1) * self.shrink_ratio + 1
            max_length = torch.max(lengths).item()
            dummy_mask = torch.arange(max_length, device=codes.device).type_as(lengths).expand(len(lengths), max_length)
            dummy_mask = dummy_mask < lengths.unsqueeze(1)
            deconv_output = self.text_conv_encoder(quantize.transpose(1, 2), lengths, pad_num=0,
                                                   mask=dummy_mask)
            quantize = deconv_output.transpose(1, 2)[:, :max_length, :]  # B x T x C

        quantize_out = {'encoder_out': quantize.transpose(0, 1),  # masked T X batch x C
                        'encoder_padding_mask': code_mask,  # B x T, this mask sets padding to be True
                        }

        return quantize_out


@register_model_architecture('vqvae_lm', 'vqvae_lm')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.bottom_conv_kernel_size = getattr(args, 'bottom_conv_kernel_size', '5,3')
    args.bottom_conv_stride = getattr(args, 'bottom_conv_stride', '2,2')
    args.bottom_latent_dim = getattr(args, 'bottom_latent_dim', args.encoder_embed_dim)
    args.bottom_latent_k = getattr(args, 'bottom_latent_k', 4096)

    args.soft_em = getattr(args, 'soft_em', 0)
    args.soft_max_temp = getattr(args, 'soft_max_temp', 5.)
    args.soft_min_temp = getattr(args, 'soft_min_temp', 1.)
    args.soft_temp_anneal_steps = getattr(args, 'soft_temp_anneal_steps', 0)
    args.soft_samples = getattr(args, 'soft_samples', 5)

    args.use_bottom_quantants_encoder = getattr(args, 'use_bottom_quantants_encoder', 0)
    args.bottom_encoder_ffn_embed_dim = getattr(args, 'bottom_encoder_ffn_embed_dim', 1024)
    args.bottom_encoder_layers = getattr(args, 'bottom_encoder_layers', 3)
    args.bottom_encoder_attention_heads = getattr(args, 'bottom_encoder_attention_heads', 4)

    args.drop_word_prob = getattr(args, 'drop_word_prob', 0.0)
    args.use_latent = getattr(args, 'use_latent', 1)
    args.pretrain_steps = getattr(args, 'pretrain_steps', -1)
    args.add_latent_positions = getattr(args, 'add_latent_positions', 0)
    args.context_window = getattr(args, 'context_window', 0)

    args.use_global_quantant = getattr(args, 'use_global_quantant', 0)
    args.global_latent_dim = getattr(args, 'global_latent_dim', args.bottom_latent_dim)
    args.global_latent_k = getattr(args, 'global_latent_k', args.bottom_latent_k // 8)


@register_model_architecture('vqvae_lm', "vqvae_lm_base")
def vqvae_base(args):
    base_architecture(args)
