"""Intermediate-geometry prespecified intermediate-geometry diagnostics; keeps old results intact."""
import hashlib
import json
import math
from pathlib import Path
import extended_evaluation as evaluation

ROOT = Path(__file__).resolve().parent

def main():
    targets = {}
    for lam in (0.25, 1.0):
        theta = math.asin(math.sqrt(lam / 6))
        targets[f'lambda_{lam:g}'] = [[math.sin(theta), 0.0, math.cos(theta)]] * 6
    design = {'design_version': 'intermediate_geometry-intermediate-v1', 'targets_per_n': {'6': targets},
              'planned': 'All eight frozen comparators; n=6, N=4096, corr07, 10000 trials, epsilon=.05, delta=1/3. No tuning or row selection.'}
    design_path = ROOT.parent / "data" / "intermediate_geometry_design.json"
    design_path.write_text(json.dumps(design, indent=2) + '\n')
    # Configure the existing evaluation driver in this process only. Production
    # estimators, scoring, selector tolerance and historical files are unchanged.
    evaluation.DESIGN_PATH = design_path
    evaluation.FROZEN_FILE_HASH = hashlib.sha256(design_path.read_bytes()).hexdigest()
    evaluation.N_LIST = [6]
    evaluation.TIDS = list(targets)
    evaluation.SEED_NEW = 20260926
    result = evaluation.run_extended_evaluation()
    result['label'] = 'Intermediate-geometry intermediate geometry diagnostics'
    result['scope'] = design['planned']
    result['command'] = 'python code/intermediate_geometry_evaluation.py'
    result['source_hashes']['intermediate_geometry_evaluation.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT.parent / "results" / "intermediate_geometry_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    assert result['primary_ok'] == 16 and result['primary_errors'] == 0

if __name__ == '__main__':
    main()
