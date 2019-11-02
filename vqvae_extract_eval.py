#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os, io
from fairseq import checkpoint_utils, options, progress_bar, utils, bleu
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.data import data_utils


def set_up_model(args, path, override_args=None):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None

    # Load ensemble
    print('| loading model(s) from {}'.format(path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [path],
        arg_overrides=overrides,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    print(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    generator = task.build_generator(model_args)
    return model, task, criterion, generator


def main(args, override_args=None):
    utils.import_user_module(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    eval_task = args.eval_task
    model, task, criterion, generator = set_up_model(args, args.vqvae_path, override_args)
    if eval_task == 'sampling':
        assert args.prior_path is not None
        prior_model, prior_task, prior_criterion, prior_generator = set_up_model(args, args.prior_path, override_args)

    dictionary = task.dictionary
    if eval_task == 'code_extract':
        fopt = io.open(os.path.join(args.results_path, args.valid_subset + ".codes"), "w", encoding='utf-8')
    elif eval_task == 'reconstruct':
        fopt = io.open(os.path.join(args.results_path, args.valid_subset + ".reconstruction"), "w", encoding='utf-8')
        # Generate and compute BLEU score
        if args.sacrebleu:
            scorer = bleu.SacrebleuScorer()
        else:
            scorer = bleu.Scorer(dictionary.pad(), dictionary.eos(), dictionary.unk())
    elif eval_task == 'sampling':
        fopt = io.open(os.path.join(args.results_path, args.valid_subset + ".samples"), "w", encoding='utf-8')
    else:
        raise ValueError

    # Initialize generator
    gen_timer = StopwatchMeter()
    num_sentences = 0

    if eval_task != 'sampling':
        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for subset in args.valid_subset.split(','):
            try:
                task.load_dataset(subset, combine=False, epoch=0)
                dataset = task.dataset(subset)
            except KeyError:
                raise Exception('Cannot find dataset: ' + subset)

            # Initialize data iterator
            itr = task.get_batch_iterator(
                dataset=dataset,
                max_tokens=args.max_tokens,
                max_sentences=args.max_sentences,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    model.max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_workers=args.num_workers,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.build_progress_bar(
                args, itr,
                prefix='valid on \'{}\' subset'.format(subset),
                no_progress_bar='simple'
            )

            log_outputs = []
            wps_meter = TimeMeter()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                log_output = {'sample_size': sample['target'].size(0)}
                log_outputs.append(log_output)
                num_sentences += sample['nsentences']

                if eval_task == 'code_extract':
                    codes = task.extract_codes(sample, model).cpu().numpy()
                elif eval_task == 'reconstruct':
                    prefix_tokens = None
                    gen_timer.start()
                    hypos = task.reconstruct(sample, model, generator, prefix_tokens)
                    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                    gen_timer.stop(num_generated_tokens)
                    wps_meter.update(num_generated_tokens)
                    progress.log({'wps': round(wps_meter.avg)})
                else:
                    raise NotImplementedError

                progress.log(log_output, step=i)

                for i, sample_id in enumerate(sample['id'].tolist()):
                    tokens = utils.strip_pad(sample['target'][i, :], dictionary.pad())
                    origin_string = dictionary.string(tokens, bpe_symbol=args.remove_bpe, escape_unk=True)
                    if len(tokens) <= 1:
                        continue
                    bpe_string = dictionary.string(tokens, bpe_symbol=None, escape_unk=True)
                    fopt.write('T-ori-{}\t{}\n'.format(sample_id, origin_string))

                    if eval_task == 'code_extract':
                        code = codes[i]
                        fopt.write('T-bpe-{}\t{}\n'.format(sample_id, bpe_string))
                        fopt.write('C-{}\t{}\n'.format(sample_id,
                                                       ' '.join([str(x) for x in code.tolist() if x != -1])))
                    elif eval_task == 'reconstruct':
                        for j, hypo in enumerate(hypos[i][:args.nbest]):
                            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                                hypo_tokens=hypo['tokens'].int().cpu(),
                                src_str="",
                                alignment=hypo['alignment'],
                                align_dict=None,
                                tgt_dict=dictionary,
                                remove_bpe=args.remove_bpe,
                            )
                            fopt.write('H-{}\t{}\t{}\n'.format(sample_id, hypo['score'], hypo_str))
                            if args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                tokens = dictionary.encode_line(origin_string, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(origin_string, hypo_str)
                            else:
                                scorer.add(tokens, hypo_tokens)
                    else:
                        raise NotImplementedError

                if i % 100000 == 0:
                    print("Processed {} lines!".format(i))
            progress.print(log_outputs[0], tag=subset, step=i)
        if eval_task == 'reconstruct':
            print('| Reconstructed {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
                num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
            print('| Reconstruct {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    else:
        batch_size = 3072 // args.max_len_b
        gen_epochs = args.num_samples // batch_size
        latent_dictionary = prior_task.dictionary
        generate_id = 0
        for ii in range(gen_epochs):
            dummy_tokens = torch.ones(args.max_len_b, batch_size).long().cuda()
            dummy_lengths = (torch.ones(args.max_len_b) * args.max_len_b).long().cuda()
            dummy_samples = {
                        'net_input': {'src_tokens': dummy_tokens,
                                      'src_lengths': dummy_lengths,
                                     },
                        'target': dummy_tokens
                        }
            prefix_tokens = None
            code_hypos = prior_task.inference_step(prior_generator, [prior_model], dummy_samples, prefix_tokens)
            predictions = []
            for jj in range(batch_size):
                code_hypo = code_hypos[jj][0]  # best output
                latent_hypo_tokens, latent_hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=code_hypo['tokens'].int().cpu(),
                    src_str="",
                    alignment=code_hypo['alignment'],
                    align_dict=None,
                    tgt_dict=latent_dictionary,
                    remove_bpe=None,
                )
                # should have no pad and eos
                predictions.append([int(ss) for ss in latent_hypo_str.strip().split()])
            predictions = torch.LongTensor(predictions).cuda()
            merged_codes = data_utils.collate_tokens(
                predictions, latent_dictionary.pad(), latent_dictionary.eos(), left_pad=False,
            )
            code_masks = merged_codes.eq(latent_dictionary.pad())

            hypos = task.sampling(dummy_samples, merged_codes, code_masks, model, generator)
            for tt in range(len(hypos)):
                hypo = hypos[tt][0]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str="",
                    alignment=hypo['alignment'],
                    align_dict=None,
                    tgt_dict=dictionary,
                    remove_bpe=args.remove_bpe,
                )
                fopt.write('H-{}\t{}\t{}\n'.format(generate_id, hypo['score'], hypo_str))
                generate_id += 1

            if generate_id % 1000 == 0:
                print("Sampled {} sentences!".format(generate_id))
    fopt.close()


def cli_main():
    parser = options.get_eval_vqvae_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_eval_vqvae_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    main(args, override_args)


if __name__ == '__main__':
    cli_main()
