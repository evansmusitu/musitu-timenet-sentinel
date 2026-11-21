"""Past-only telemetry and grouped splits for OpenTime binary question answering."""
import numpy as np
from .data import COLS

QUESTION = 'Will this simulated engine reach end of useful life within the next 30 operating cycles? Answer Yes or No.'

def partitions(frame, seed=42):
    ids = np.array(sorted(frame.unit.unique()))
    if len(ids)<2:
        raise ValueError('At least two development engines required')
    np.random.default_rng(seed).shuffle(ids)
    cut=max(1,min(len(ids)-1,int(.8*len(ids))))
    return frame[frame.unit.isin(ids[:cut])],frame[frame.unit.isin(ids[cut:])]

def examples(frame, *, official_rul=None, window=30, stride=15):
    if window<2 or stride<1:
        raise ValueError('Invalid window/stride')
    out=[]
    for unit,g in frame.groupby('unit',sort=True):
        g=g.sort_values('cycle').reset_index(drop=True)
        if len(g)<window:
            if official_rul is not None:
                raise ValueError('Official engine history too short')
            continue
        ends=[len(g)-1] if official_rul is not None else range(window-1,len(g),stride)
        for end in ends:
            cycle=int(g.cycle.iloc[end]); w=g.iloc[end-window+1:end+1]
            rul=int(official_rul[unit]) if official_rul is not None else int(g.cycle.iloc[-1])-cycle
            out.append({'unit':int(unit),'cycle':cycle,'answer':'Yes' if rul<=30 else 'No',
                        'series':[{'name':c,'values':w[c].astype(float).tolist()} for c in COLS[2:]]})
    return out
