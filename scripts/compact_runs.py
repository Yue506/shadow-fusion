"""Create same-image comparisons across existing batch runs."""
import argparse
from shadow_fusion.benchmark import load_manifest, save_compact

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--output-dir', required=True)
args = p.parse_args()
save_compact(load_manifest(args.manifest), args.runs, args.output_dir)
