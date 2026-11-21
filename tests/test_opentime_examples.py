import pandas as pd
from sentinel.data import COLS
from sentinel.opentime_examples import examples, partitions

def frame(units, length=45):
    return pd.DataFrame([{'unit': u, 'cycle': t, **{c: float(u*100+t) for c in COLS[2:]}} for u in units for t in range(1,length+1)])

def test_past_only_and_official_endpoints():
    dev=frame([1,2]); official=frame([1,2]); labels={1:12,2:55}
    items=examples(dev, window=30, stride=15)
    assert items[0]['series'][0]['values']==list(map(float,range(101,131)))
    assert items[0]['answer']=='Yes'
    mutated=dev.copy(); mutated.loc[mutated.cycle>30,'sensor_1']=99999
    assert items[0]['series']==examples(mutated,window=30,stride=15)[0]['series']
    out=examples(official, official_rul=labels)
    assert [(x['unit'],x['cycle'],x['answer']) for x in out]==[(1,45,'Yes'),(2,45,'No')]

def test_engine_partitions_disjoint():
    tr,va=partitions(frame(list(range(1,11))))
    assert set(tr.unit).isdisjoint(set(va.unit))
    assert set(tr.unit)|set(va.unit)==set(range(1,11))
