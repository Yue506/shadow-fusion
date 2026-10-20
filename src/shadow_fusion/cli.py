import argparse
import json
import sys
from pathlib import Path
from .pipeline import MODULES, PRESETS, make_config, compose
from .io import read_rgb, read_mask, sha256, write_result


def config_from_args(args):
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    if not isinstance(values, dict) or set(values)-{'mode', *MODULES}:
        raise ValueError('Config JSON must contain only mode, color, extra, gamma, feather')
    mode = args.mode or values.pop('mode', 'shadow')
    values.pop('mode', None)
    for key in ('color', 'extra', 'gamma'):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key) == 'on'
    if args.feather is not None:
        values['feather'] = args.feather
    ablations = tuple(x.strip() for x in args.ablate.split(',') if x.strip())
    return make_config(mode, ablate=ablations, **values), ablations


def parser():
    p = argparse.ArgumentParser(description='Shadow-aware image compositing; module ablations are explicit.')
    commands = p.add_subparsers(dest='command', required=True)
    single = commands.add_parser('compose', help='Compose one source/candidate/mask triplet')
    single.add_argument('--source', required=True)
    single.add_argument('--generated', required=True)
    single.add_argument('--mask', required=True)
    single.add_argument('--output', required=True)
    single.add_argument('--artifacts-dir')
    single.add_argument('--overwrite', action='store_true')
    batch = commands.add_parser('batch', help='Run one configuration over a JSONL dataset')
    batch.add_argument('--manifest', required=True)
    batch.add_argument('--output-dir', required=True)
    for command in (single, batch):
        command.add_argument('--config', help='JSON configuration; explicit flags override JSON')
        command.add_argument('--mode', choices=tuple(PRESETS))
        for key in ('color', 'extra', 'gamma'):
            command.add_argument('--'+key, choices=('on', 'off'))
        command.add_argument('--feather', choices=('none', 'single', 'multilevel', 'paper'))
        command.add_argument('--ablate', default='', help='Comma-separated: color,extra,gamma,feather. Ablations take precedence.')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        config, ablations = config_from_args(args)
        if args.command == 'batch':
            from .benchmark import run_batch
            result = run_batch(args.manifest, args.output_dir, config, ablations)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        inputs = [Path(getattr(args, k)).resolve() for k in ('source', 'generated', 'mask')]
        if Path(args.output).resolve() in inputs:
            raise ValueError('Output must not overwrite an input image')
        source, profile = read_rgb(args.source)
        generated, _ = read_rgb(args.generated)
        result = compose(source, generated, read_mask(args.mask), config)
        result['metadata'].update(ablations=list(ablations),
                                  input_sha256={key: sha256(getattr(args, key)) for key in ('source', 'generated', 'mask')})
        metadata = write_result(result, args.output, overwrite=args.overwrite,
                                icc_profile=profile, artifacts_dir=args.artifacts_dir)
        print(json.dumps({'output': str(Path(args.output).resolve()), 'metadata': str(metadata.resolve()),
                          'config': result['metadata']['config']}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        p.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
