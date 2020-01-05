#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import os, io
import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)
        if args.best_checkpoint_metric == 'bleu' and not os.path.exists(args.eval_dir):
            os.mkdir(args.eval_dir)
            args.remove_bpe = '@@ '

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Build generator if evaluate with BLEU score
    if args.best_checkpoint_metric == 'bleu':
        generator = task.build_generator(args)
    else:
        generator = None

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr, generator)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            if args.best_checkpoint_metric == 'bleu':
                valid_bleu = multi_gpu_bleu(args, trainer, task, generator, trainer._model, epoch_itr, valid_subsets,
                                              pprefix="valid", valid_bleu=-1, log=False)
                test_bleu = multi_gpu_bleu(args, trainer, task, generator, trainer._model, epoch_itr, ['test'],
                                           pprefix="test", valid_bleu=valid_bleu, log=True)
                valid_losses = [valid_bleu]
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr, generator=None):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            elif 'true_nll_loss' in k:
                extra_meters[k].update(v, log_output['ntokens'])
                stats['true_ppl'] = utils.get_perplexity(extra_meters[k].avg)
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            if args.best_checkpoint_metric == 'bleu':
                valid_bleu = multi_gpu_bleu(args, trainer, task, generator, trainer._model, epoch_itr, valid_subsets,
                                              pprefix="valid", valid_bleu=-1, log=False)
                test_bleu = multi_gpu_bleu(args, trainer, task, generator, trainer._model, epoch_itr, ['test'],
                                           pprefix="test", valid_bleu=valid_bleu, log=True)
                valid_losses = [valid_bleu]

            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        if k == "true_nll_loss":
            stats['true_ppl'] = utils.get_perplexity(meter.avg)
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=0,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                if k == 'true_nll_loss':
                    extra_meters[k].update(v, log_output['ntokens'])
                else:
                    extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            if k == "true_nll_loss":
                stats['true_ppl'] = utils.get_perplexity(meter.avg)
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        # valid_losses.append(
        #     stats[args.best_checkpoint_metric].avg
        #     if args.best_checkpoint_metric == 'loss'
        #     else stats[args.best_checkpoint_metric]
        # )
        valid_losses.append(stats['loss'].avg)
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best') and (not args.best_checkpoint_metric == 'bleu'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def log_test(bleu, valid_bleu, trainer, progress):
    stats = collections.OrderedDict()
    if (not hasattr(checkpoint_utils.save_checkpoint, 'prev_best')) or (
            hasattr(checkpoint_utils.save_checkpoint, 'prev_best') and valid_bleu > checkpoint_utils.save_checkpoint.prev_best):
        trainer.meters['best_test_bleu_at_best_valid'] = bleu
        stats['best_valid_bleu'] = valid_bleu
    else:
        stats['best_valid_bleu'] = checkpoint_utils.save_checkpoint.prev_best

    key = "best_test_bleu"
    if bleu > trainer.meters[key] or key not in trainer.meters:
        trainer.meters[key] = bleu

    stats['valid_bleu'] = valid_bleu
    stats['test_bleu'] = bleu
    stats[key] = trainer.get_meter('best_test_bleu')
    stats['best_test_bleu_at_best_valid'] = trainer.get_meter("best_test_bleu_at_best_valid")
    progress.print(stats, tag='test', step=trainer.get_num_updates())


def multi_gpu_bleu(args, trainer, task, generator, model, epoch_itr, subsets, pprefix="valid", valid_bleu=None, log=False):
    def calc_bleu(fref, fmt, result_path):
        current_path = os.path.dirname(os.path.realpath(__file__))
        script = os.path.join(current_path, 'scripts/multi-bleu.perl')
        temp = os.path.join(result_path, 'tmp')
        os.system("perl %s %s < %s > %s" % (script, fref, fmt, temp))
        bleu = open(temp, 'r').read().strip()
        bleu = bleu.split(",")[0].split("=")
        if len(bleu) < 2:
            return 0.0
        bleu = float(bleu[1].strip())
        return bleu

    epoch = epoch_itr.epoch
    updates = trainer.get_num_updates()
    ftran_filename = os.path.join(args.eval_dir, '{}_{}_{}.translation'.format(pprefix, epoch, updates))
    fgold_filename = os.path.join(args.eval_dir, '{}_{}_{}.gold'.format(pprefix, epoch, updates))

    def merge(prefix, total_ranks):
        filelists = []
        for i in range(total_ranks):
            filelists.append(os.path.join(args.eval_dir, '{}_{}.translation'.format(prefix, i)))
            filelists.append(os.path.join(args.eval_dir, '{}_{}.gold'.format(prefix, i)))

        new_ftran_filename = os.path.join(args.eval_dir, '{}_{}.translation'.format(epoch, updates))
        new_fgold_filename = os.path.join(args.eval_dir, '{}_{}.gold'.format(epoch, updates))

        with io.open(new_ftran_filename, "w", encoding="utf-8") as ft, io.open(new_fgold_filename, "w", encoding="utf-8") as fg:
            for i in range(total_ranks):
                with io.open(os.path.join(args.eval_dir, '{}_{}.translation'.format(prefix, i)), "r", encoding="utf-8") as fti, \
                    io.open(os.path.join(args.eval_dir, '{}_{}.gold'.format(prefix, i)), "r",
                            encoding="utf-8") as fgi:
                    for lt, lg in zip(fti, fgi):
                        ft.write(lt.strip() + "\n")
                        fg.write(lg.strip() + "\n")

        for path in filelists:
            os.remove(path)
        return new_ftran_filename, new_fgold_filename

    def write_to_file(ftran, fgold, trans, golds):
        for t, g in zip(trans, golds):
            ftran.write(t.strip() + "\n")
            fgold.write(g.strip() + "\n")

    if args.distributed_world_size > 1:
        ftran_filename = os.path.join(args.eval_dir, '{}_{}_{}_{}.translation'.format(pprefix, epoch, updates, args.distributed_rank))
        fgold_filename = os.path.join(args.eval_dir, '{}_{}_{}_{}.gold'.format(pprefix, epoch, updates, args.distributed_rank))

    for subset in subsets:
        ftran = io.open(ftran_filename, "w", encoding="utf-8")
        fgold = io.open(fgold_filename, "w", encoding="utf-8")
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens + 6184,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(fix_batches_to_gpus=True, shuffle=False)

        progress = itr
        if distributed_utils.is_master(args):
            progress = progress_bar.build_progress_bar(
                args, itr, epoch_itr.epoch,
                prefix=f'test on \'{subset}\' subset',
                no_progress_bar='simple'
            )

        ignore_results = False
        for sample in progress:
            if sample is None or len(sample) == 0:
                sample = trainer._dummy_batch
                ignore_results = True
            sample = trainer._prepare_sample(sample)
            hypos = task.inference_step(generator, [model], sample, prefix_tokens=None)

            if not ignore_results:
                hypo_strings = [task.tgt_dict.string(hypo[:1]['tokens'].int().cpu(), args.remove_bpe) for hypo in hypos]
                target_tokens = [utils.strip_pad(tt, task.tgt_dict.pad()).int().cpu() for tt in sample['target']]
                tgt_strings = [task.tgt_dict.string(tokens, args.remove_bpe, escape_unk=True) for tokens in target_tokens]
                write_to_file(ftran, fgold, hypo_strings, tgt_strings)

        ftran.close()
        fgold.close()

        completed = True
        if args.distributed_world_size > 1:
            completed = distributed_utils.all_gather_list(completed)

        if distributed_utils.is_master(args):
            prefix = "{}_{}_{}".format(pprefix, epoch, updates)
            ftran_filename, fgold_filename = merge(prefix, args.distributed_world_size)
            bleu = calc_bleu(ftran_filename, fgold_filename, args.eval_dir)
            if log:
                log_test(bleu, valid_bleu, trainer, progress)
        else:
            bleu = None

    return bleu


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
