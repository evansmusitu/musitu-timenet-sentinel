"""Build an aggregate-only simulated domain-shift lab from verified run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from string import Template

SOURCE_MD5="79a22f36e80606c69d0e9e4da5bb2b7a"
SUBSETS=("FD001","FD002","FD003","FD004")
PROOF_CLASSICAL="https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003875499"
PROOF_COMPACT="https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45003991598"
PROOF_MULTIDOMAIN="https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45004350622"
PROOF_CLASSIFIER="https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/45004601674"


def validate(baseline: dict,compact: dict,multidomain: dict,classifier:dict)->None:
    if any(d.get('source_md5')!=SOURCE_MD5 for d in (baseline,compact,multidomain,classifier)):
        raise ValueError('Source digest mismatch')
    if not baseline.get('reference_fd001_reproduced') or not compact.get('reference_fd001_checkpoint_reproduced'):
        raise ValueError('Original source models were not reproduced')
    if baseline.get('training_cohort_only')!='FD001':raise ValueError('Classical source model identity changed')
    if 'NOT OpenTime 5B' not in compact.get('model_identity',''):
        raise ValueError('Compact model identity is not distinguished from OpenTime 5B')
    b=baseline['official_test_evaluations'];c=compact['official_test_evaluations']
    if set(b)!=set(SUBSETS) or set(c)!=set(SUBSETS):raise ValueError('Incomplete simulated cohorts')
    for cohort in SUBSETS:
        bb,cc=b[cohort],c[cohort]
        for field in ['heldout_test_engines','official_test_engines_total','excluded_due_to_fewer_than_30_observations']:
            if bb[field]!=cc[field]:raise ValueError('Test engine coverage mismatch in '+cohort)
        for v in (bb,cc):
            if sum(v['confusion'].values())!=v['heldout_test_engines']:
                raise ValueError('Confusion counts inconsistent with held-out endpoints')
            if v['heldout_test_engines']+v['excluded_due_to_fewer_than_30_observations']!=v['official_test_engines_total']:
                raise ValueError('Short-history exclusions not accounted for')
    if b['FD001']['heldout_test_engines']!=100:raise ValueError('FD001 source population incomplete')
    if compact['official_test_evaluations']['FD001']['trained_warning_head']['f1']<0.9599:
        raise ValueError('Compact FD001 original F1 did not reproduce')
    if multidomain.get('training_cohorts')!=['FD001','FD002','FD003'] or multidomain.get('heldout_target_cohort')!='FD004':
        raise ValueError('Multidomain fit or target changed')
    if multidomain.get('fd004_data_used_in_fit_or_validation') is not False:
        raise ValueError('FD004 leakage')
    if multidomain['official_fd004_heldout_test']['heldout_test_engines']!=b['FD004']['heldout_test_engines']:
        raise ValueError('FD004 sample mismatch')
    if classifier.get('target_fd004_data_used_to_fit_or_select_threshold') is not False:
        raise ValueError('Classifier selected threshold from official test data')
    if classifier.get('training_only_cohorts')!=['FD001','FD002','FD003'] or classifier.get('heldout_test_cohort')!='FD004':
        raise ValueError('Classifier training/holdout cohort changed')
    ct=classifier['official_fd004_test']
    if ct['eligible_engines']!=b['FD004']['heldout_test_engines'] or ct['total_official_test_engines']!=b['FD004']['official_test_engines_total']:
        raise ValueError('Classifier uses different official FD004 population')
    if sum(ct[k] for k in ['tp','fp','fn','tn'])!=ct['eligible_engines']:
        raise ValueError('Classifier confusion matrix invalid')


def build_aggregate(baseline,compact,multidomain,classifier)->dict:
    validate(baseline,compact,multidomain,classifier)
    result={"project":"MUSITU AXIOM Sentinel","scientific_scope":"NASA simulated engine datasets, never actual OEM validation",
            "confidence_note":"Post-hoc exploratory domain-shift analysis; negative transfer is fully reported",
            "official_dataset_source":"https://zenodo.org/records/15346912","source_md5":SOURCE_MD5,
            "provenance":{"original_classical":PROOF_CLASSICAL,"original_compact":PROOF_COMPACT,"exploratory_multidomain":PROOF_MULTIDOMAIN,"exploratory_classifier":PROOF_CLASSIFIER},
            "rows":[],"multi_domain_fd004":multidomain['official_fd004_heldout_test'],
            "multi_domain_warning_classifier_fd004":classifier["official_fd004_test"],
            "classifier_validation_threshold":classifier["chosen_validation_threshold"],
            "experimental_multi_domain_training_sources":multidomain['training_cohorts'],
            "limitations":["All sets are NASA simulations, not real aircraft or wind turbines.",
                 "FD001/FD002/FD003 training differs from the organizer's OpenTime 5B checkpoint; all results on this page are separate CPU models.",
                 "FD001-only frozen models were not retuned on test sets; shifted data expose real generalization failures.",
                 "Short test histories under 30 cycles were excluded and visibly counted.",
                 "Exploratory FD001+FD002+FD003 to FD004 training and the dedicated warning classifier were decided after detecting shift failures; neither is preregistered.",
                 "Dataset metadata lacks a sufficiently specific source SPDX license; aggregate evaluation only, no raw rows or model weights redistributed."]}
    for k,description in [('FD001','Familiar conditions and one fault mode'),('FD002','Six conditions, one fault mode'),
                          ('FD003','One condition, two fault modes'),('FD004','Six conditions, two fault modes')]:
        b=baseline['official_test_evaluations'][k];c=compact['official_test_evaluations'][k]
        result['rows'].append({'cohort':k,'description':description,'n':b['heldout_test_engines'],
           'official_total':b['official_test_engines_total'],
           'short_excluded':b['excluded_due_to_fewer_than_30_observations'],
           'classical_f1':b['warning_30_f1'],'compact_f1':c['trained_warning_head']['f1'],
           'classical_mae':b['rul_mae_cycles'],'compact_mae':c['rul_mae_cycles'],
           'classical_confusion':b['confusion'],'compact_confusion':c['confusion']})
    return result


def render(report:dict,template:Path)->str:
    bars=[];table=[]
    for r in report['rows']:
        b=float(r['classical_f1']);c=float(r['compact_f1']);k=html.escape(r['cohort']);d=html.escape(r['description'])
        bars.append(f'<div class="bar-row"><div class="bar-label"><strong>{k}</strong><span>{d}</span></div>'
          f'<div class="rail"><div class="bar classical" style="width:{b*100:.1f}%"></div></div><span class="bar-value">{b:.3f}</span>'
          f'<div class="rail"><div class="bar compact" style="width:{c*100:.1f}%"></div></div><span class="bar-value">{c:.3f}</span></div>')
        cb=r['compact_confusion']
        table.append(f'<tr><th scope="row">{k}<small>{d}</small></th><td>{r["n"]} / {r["official_total"]}</td>'
             f'<td>{b:.3f}</td><td>{c:.3f}</td><td>{r["classical_mae"]:.2f}</td>'
             f'<td>{r["compact_mae"]:.2f}</td><td>{cb["fp"]}</td><td>{cb["fn"]}</td></tr>')
    m=report['multi_domain_fd004'];clf=report['multi_domain_warning_classifier_fd004'];bline=report['rows'][-1]
    tmpl=Template(template.read_text(encoding='utf-8'))
    page=tmpl.substitute(bars='\n'.join(bars),rows='\n'.join(table),
       baseline_fd004_f1=f'{bline["classical_f1"]:.3f}',
       multidomain_fd004_f1=f'{m["warning_30_f1"]:.3f}',
       multidomain_fd004_mae=f'{m["rul_mae_cycles"]:.2f}',
       multidomain_fd004_tp=str(m['confusion']['tp']),
       multidomain_fd004_fp=str(m['confusion']['fp']),
       multidomain_fd004_fn=str(m['confusion']['fn']),
       multidomain_fd004_tn=str(m['confusion']['tn']),
       compact_fd004_fp=str(bline['compact_confusion']['fp']),
       compact_fd004_fn=str(bline['compact_confusion']['fn']),
       classifier_fd004_f1=f"{clf['f1']:.3f}",
       classifier_fd004_precision=f"{clf['precision']*100:.1f}%",
       classifier_fd004_recall=f"{clf['recall']*100:.1f}%",
       classifier_fd004_tp=str(clf['tp']), classifier_fd004_fp=str(clf['fp']),
       classifier_fd004_fn=str(clf['fn']),classifier_fd004_tn=str(clf['tn']),
       classifier_threshold=f"{report['classifier_validation_threshold']:.2f}")
    return page


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--compact',type=Path,required=True)
    p.add_argument('--multidomain',type=Path,required=True)
    p.add_argument('--classifier',type=Path,required=True)
    p.add_argument('--template',type=Path,default=Path('docs/robustness-lab.template.html'))
    p.add_argument('--output',type=Path,default=Path('docs/robustness-lab.html'))
    p.add_argument('--json',type=Path,default=Path('docs/evidence/cross_cohort_robustness_audit.json'))
    a=p.parse_args()
    b=json.loads(a.baseline.read_text());c=json.loads(a.compact.read_text());m=json.loads(a.multidomain.read_text())
    clf=json.loads(a.classifier.read_text())
    report=build_aggregate(b,c,m,clf)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.json.parent.mkdir(parents=True,exist_ok=True)
    a.json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    a.output.write_text(render(report,a.template),encoding='utf-8')
    print('Rendered',a.output, 'sha256', hashlib.sha256(a.output.read_bytes()).hexdigest())

if __name__=='__main__':main()
