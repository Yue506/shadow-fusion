"""Freeze one complete configuration from validation evidence, before final test."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
from shadow_fusion import make_config


def choose(scores,rules):
    eligible=[s for s in scores if s['min_stability']>=rules['stability_guard']]
    passed=bool(eligible)
    if not eligible:eligible=scores
    best=max(s['min_f1'] for s in eligible)
    band=[s for s in eligible if s['min_f1']>=best-rules['f1_band']]
    winner=max(band,key=lambda s:(s['min_recall'],-s['max_background'],s['min_stability'],-s['mean_seconds']))
    return winner,passed


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(roots,plan_path,rules_path,selection_path,final_plan_path,output_config):
    targets=[Path(selection_path),Path(final_plan_path),Path(output_config)]
    if any(p.exists() for p in targets):raise FileExistsError('Selection artifacts are immutable; inspect the existing selection instead of replacing it')
    plan=json.loads(Path(plan_path).read_text());rules=json.loads(Path(rules_path).read_text())
    evidence=[]
    for root in map(Path,roots):
        payload=json.loads((root/'metrics.json').read_text())
        cohort=json.loads((root/'progress.jsonl').read_text().splitlines()[0])['cohort']
        if payload['count']!=20 or payload['plan_sha256']!=sha(plan_path):raise ValueError('Validation evidence count or plan hash mismatch')
        evidence.append((cohort,payload,root))
    if sorted(c for c,_,_ in evidence)!=sorted(rules['cohorts']):raise ValueError('Both fixed candidate conditions are required')
    scores=[]
    for name in rules['selection_ids']:
        summaries=[p['methods'][name]['summary'] for _,p,_ in evidence]
        scores.append({'id':name,'min_f1':min(s['shadow_f1'] for s in summaries),
                       'min_recall':min(s['shadow_recall'] for s in summaries),
                       'min_stability':min(s['stability_iou_mean'] for s in summaries),
                       'max_background':max(s['background_changed_fraction'] for s in summaries),
                       'mean_seconds':sum(s['cpu_seconds'] for s in summaries)/len(summaries)})
    winner,guard=choose(scores,rules)
    selected=next(p for p in plan if p['id']==winner['id'])
    final=[{'id':mode,'engine':'baseline','config':asdict(make_config(mode))} for mode in ['mask','tracefusion','shadow']]
    final.append({'id':'selected','engine':selected['engine'],'config':selected['config']})
    final_text=json.dumps(final,indent=2)+'\n'
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_before_test':True,
             'selection_code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'selected_validation_id':winner['id'],'selected_config':selected['config'],
             'stability_guard_passed':guard,'validation_plan_sha256':sha(plan_path),
             'rules_sha256':sha(rules_path),'plan_sha256':hashlib.sha256(final_text.encode()).hexdigest(),
             'validation_evidence':{co:{'metrics_sha256':sha(root/'metrics.json'),'manifest_sha256':p['manifest_sha256']} for co,p,root in evidence},
             'candidate_scores':scores,'rule':rules['ranking'],
             'quality_scope':'Validation selection for reference-derived proxy conditions; semantic background identity is not guaranteed.'}
    for path in targets:path.parent.mkdir(parents=True,exist_ok=True)
    targets[0].write_text(json.dumps(payload,indent=2)+'\n')
    targets[1].write_text(final_text)
    targets[2].write_text(json.dumps(selected['config'],indent=2)+'\n')
    print(json.dumps({'winner':winner,'guard_passed':guard,'selection':str(targets[0]),'final_plan':str(targets[1])},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation-roots',nargs=2,required=True)
    p.add_argument('--plan',default='configs/campaign_validation.json');p.add_argument('--rules',default='docs/results/campaign_20261021/selection_rules.json')
    p.add_argument('--selection',default='docs/results/campaign_20261021/selection.json')
    p.add_argument('--final-plan',default='configs/campaign_final.json');p.add_argument('--output-config',default='configs/selected.json')
    a=p.parse_args();select(a.validation_roots,a.plan,a.rules,a.selection,a.final_plan,a.output_config)
