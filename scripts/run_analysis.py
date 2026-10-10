'''Measure galaxy shapes and infer an NFW halo mass from local coursework inputs.'''

import argparse
import json
import os
from pathlib import Path

from weak_lensing.pipeline import run_coursework


def main() -> None:
    '''Parse input locations and run the coursework analysis.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('A6_DATA_DIR'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/coursework'))
    parser.add_argument('--grid-size', type=int, default=120)
    parser.add_argument('--max-sources', type=int)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error('Provide --data-dir or set A6_DATA_DIR to the coursework input directory.')

    def progress(completed: int, total: int) -> None:
        '''Report measurement progress at approximately 10,000-source intervals.'''
        if completed == total or completed // 10000 > max(0, completed - args.batch_size) // 10000:
            print(f'Measured {completed:,} / {total:,} galaxy stamps', flush=True)

    summary = run_coursework(args.data_dir, args.output_dir, grid_size=args.grid_size,
                             max_sources=args.max_sources, batch_size=args.batch_size, progress=progress)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
