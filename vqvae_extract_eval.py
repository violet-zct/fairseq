#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os, io
from fairseq import checkpoint_utils, options, progress_bar, utils


def main(args, override_args=None):
    utils.import_user_module(args)

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
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

    fcodes = io.open(os.path.join(args.results_path, args.gen_subset + ".codes"), "w", encoding='utf-8')
    dictionary = task.dictionary

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
                *[m.max_positions() for m in models],
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
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            codes = task.extract_codes(sample, model).cpu().numpy()
            log_output = {'sample_size': sample['target'].size(0)}
            progress.log(log_output, step=i)
            log_outputs.append(log_output)


            for i, sample_id in enumerate(sample['id'].tolist()):
                tokens = utils.strip_pad(sample['target'][i, :], dictionary.pad())
                origin_string = dictionary.string(tokens, bpe_symbol=args.remove_bpe, escape_unk=True)
                bpe_string = dictionary.string(tokens, bpe_symbol=None, escape_unk=True)
                code = codes[i]
                fcodes.write('T-bpe-{}\t{}'.format(sample_id, bpe_string))
                fcodes.write('T-ori-{}\t{}'.format(sample_id, origin_string))
                fcodes.write('C-{}\t{}'.format(sample_id,
                                               ' '.join(str(x) for x in code.tolist())))
        fcodes.close()
        progress.print(log_outputs[0], tag=subset, step=i)


def cli_main():
    parser = options.get_eval_vqvae_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_eval_vqvae_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    main(args, override_args)


if __name__ == '__main__':
    cli_main()
