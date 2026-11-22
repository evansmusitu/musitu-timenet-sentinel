"""CPU-only engine-diverse sampling and split controls for a future OpenTime pilot.

Pure preparation protocol. This module never trains, evaluates on official test,
provisions cloud GPUs, changes prior metrics, or promises performance gains.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _validate(rows: list[dict]) -> None:
    seen: set[tuple[int,int]] = set()
    for item in rows:
        unit, cycle, label = item.get('unit'), item.get('cycle'), item.get('answer')
        if (not isinstance(unit,int) or isinstance(unit,bool) or unit<=0 or
                not isinstance(cycle,int) or isinstance(cycle,bool) or cycle<=0):
            raise ValueError('positive integer unit and cycle required')
        if label not in ('Yes','No'):
            raise ValueError('Each warning answer must be Yes or No')
        if (unit,cycle) in seen:
            raise ValueError('duplicate engine/cycle window')
        seen.add((unit,cycle))


def balanced_engine_diverse(rows: list[dict], limit: int, *, seed: int = 42) -> list[dict]:
    """Balance Yes/No while sampling across distinct engines before repeat windows."""
    if not isinstance(limit,int) or limit<2:
        raise ValueError('sample limit must be at least 2')
    _validate(rows)
    grouped: dict[str,dict[int,list[dict]]] = {label:defaultdict(list) for label in ('Yes','No')}
    for item in rows:
        grouped[item['answer']][item['unit']].append(item)
    if not all(grouped[label] for label in ('Yes','No')):
        raise ValueError('both Yes and No warning classes required')
    per_class=min(limit//2, *(sum(map(len,grouped[label].values())) for label in ('Yes','No')))
    rng=random.Random(seed)
    chosen=[]
    for label in ('Yes','No'):
        groups=list(grouped[label].items())
        rng.shuffle(groups)
        queues=[]
        for engine,windows in groups:
            ordered=list(windows)
            rng.shuffle(ordered)
            queues.append(ordered)
        selected=0
        while selected<per_class:
            progress=False
            for queue in queues:
                if selected>=per_class: break
                if queue:
                    chosen.append(queue.pop())
                    selected+=1
                    progress=True
            if not progress:
                raise ValueError('Insufficient class windows after validation')
    rng.shuffle(chosen)
    return chosen


def audit_engine_groups(train: list[dict], validation: list[dict]) -> dict:
    _validate(train)
    _validate(validation)
    train_ids={x['unit'] for x in train}
    val_ids={x['unit'] for x in validation}
    if train_ids.intersection(val_ids):
        raise ValueError('training and validation engine overlap detected')
    return {
        'train_unique_engines':len(train_ids),
        'validation_unique_engines':len(val_ids),
        'train_windows':len(train),
        'validation_windows':len(validation),
        'train_class_counts':dict(Counter(x['answer'] for x in train)),
        'validation_class_counts':dict(Counter(x['answer'] for x in validation)),
    }


def plan(data_dir: Path, train_limit: int, val_limit: int, *, download: bool = False) -> dict:
    """Use training data only to produce a future GPU-sampling manifest."""
    from .data import download_dataset, read_table
    from .opentime_examples import examples, partitions
    if download:
        download_dataset(data_dir)
    source=read_table(data_dir/'train_FD001.txt')
    train,validation=partitions(source)
    train_windows=examples(train)
    validation_windows=examples(validation)
    full=audit_engine_groups(train_windows,validation_windows)
    chosen_train=balanced_engine_diverse(train_windows,train_limit,seed=42)
    from .opentime_v2_sampler import prevalence_matched_engine_diverse
    chosen_val=prevalence_matched_engine_diverse(validation_windows,val_limit,seed=43)
    selected=audit_engine_groups(chosen_train,chosen_val)
    return {
      'dataset':'NASA C-MAPSS FD001 development engines only',
      'training_sampler':'balanced-engine-diverse-v1',
      'validation_sampler':'prevalence-matched-engine-diverse-v1',
      'seed':42,
      'requested_train_limit':train_limit,'requested_validation_limit':val_limit,
      'available':full,'selected':selected,
      'heldout_evaluation_performed':False,
      'checkpoint_loaded':False,
      'gpu_or_paid_compute_used':False,
      'note':'Planning only. Training classes balanced, validation prevalence matched to development source, no official-test model selection. No GPU launched or training result claimed; billing requires a real provider-enforced no-overrun guarantee.'
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    parser.add_argument('--train-limit',type=int,default=256)
    parser.add_argument('--val-limit',type=int,default=64)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('artifacts/opentime_sampling_plan.json'))
    args=parser.parse_args()
    report=plan(args.data_dir,args.train_limit,args.val_limit,download=args.download)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
