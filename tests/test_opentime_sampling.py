"""CPU-only planned OpenTime sampling protocol; not a trained-model evaluation."""
import pytest
from sentinel.opentime_sampling import balanced_engine_diverse, audit_engine_groups


def rows(n_engine=8, windows=4):
    return [{"unit": engine, "cycle": 30 + j * 15,
             "answer": "Yes" if j % 2 else "No", "series": [engine, j]}
            for engine in range(1, n_engine+1) for j in range(windows)]


def test_sampling_balances_both_classes_and_covers_engines_before_repeats():
    original=rows()
    output=balanced_engine_diverse(original, 12, seed=42)
    assert len(output)==12
    assert sum(x['answer']=='Yes' for x in output)==6
    assert sum(x['answer']=='No' for x in output)==6
    assert len({x['unit'] for x in output if x['answer']=='Yes'})==6
    assert len({x['unit'] for x in output if x['answer']=='No'})==6
    assert original == rows()
    assert output==balanced_engine_diverse(original, 12,seed=42)


def test_imbalanced_cohort_limits_pair_count_without_replacement():
    source=rows(n_engine=3,windows=4)
    subset=[x for x in source if x['answer']=='No'][:2]+[x for x in source if x['answer']=='Yes']
    output=balanced_engine_diverse(subset,50,seed=7)
    assert len(output)==4
    assert len({(x['unit'],x['cycle']) for x in output})==4


def test_no_label_or_bad_limit_is_fatal():
    with pytest.raises(ValueError,match="both"):
        balanced_engine_diverse([{'unit':1,'cycle':30,'answer':'Yes'}],8)
    with pytest.raises(ValueError,match="limit"):
        balanced_engine_diverse(rows(),1)
    with pytest.raises(ValueError,match="Yes or No"):
        balanced_engine_diverse(rows()+[{'unit':12,'cycle':30,'answer':'Maybe'}],8)


def test_duplicate_or_invalid_engine_cycle_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        balanced_engine_diverse(rows()+[rows()[0]],8)
    with pytest.raises(ValueError, match="positive"):
        balanced_engine_diverse(rows()+[{'unit':0,'cycle':30,'answer':'Yes'}],8)


def test_train_and_validation_engines_must_be_disjoint():
    train=rows(n_engine=4)
    val=[{**r,'unit':r['unit']+4} for r in rows(n_engine=4)]
    s=audit_engine_groups(train,val)
    assert s['train_unique_engines']==4 and s['validation_unique_engines']==4
    with pytest.raises(ValueError,match="overlap"):
        audit_engine_groups(train,val+[train[0]])


def test_actual_gpu_runner_sampler_prioritizes_distinct_engines():
    from sentinel.opentime_train import balanced
    source=rows(n_engine=8,windows=4)
    choice=balanced(source,12,42)
    assert len({x['unit'] for x in choice if x['answer']=='Yes'})==6
    assert len({x['unit'] for x in choice if x['answer']=='No'})==6
