"""Immutable, background-first validation selection; independent test unopened."""
import argparse
import json
from pathlib import Path
from shadow_fusion.io import sha256
from audit_background_first import audit


def select(root):
    root=Path(root);public=Path('docs/results/background_first_20261021')
    outputs=[public/'selection.json',Path('configs/background_first_selected.json'),Path('configs/campaign_background_first_final.json')]
    if any(p.exists() for p in outputs):raise FileExistsError('Selection already frozen')
    rules=json.loads((public/'selection_rules.json').read_text());assert rules['frozen_before_validation']
    plan=json.load(open('configs/campaign_background_first_validation.json'));cohorts=['target_proxy','all_shadow_proxy'];metrics={};audits={}
    for co in cohorts:
        run=root/f'val_{co}';m=json.loads((run/'metrics.json').read_text())
        assert m['count']==20 and m['plan_sha256']==sha256('configs/campaign_background_first_validation.json')
        metrics[co]=m['methods'];ap=root/f'audit_val_{co}.json'
        if not ap.exists():audit(f'data/campaign100_v3/{co}_validation.jsonl',run,ap)
        audits[co]=json.loads(ap.read_text())['summary']
    rankings=[]
    for entry in plan:
        name=entry['id'];reasons=[]
        for co in cohorts:
            s=metrics[co][name]['summary'];b=metrics[co]['conservative']['summary']
            if s['shadow_recall']<.8:reasons.append(co+': recall')
            if s['stability_iou_mean']<.95:reasons.append(co+': stability')
            if s['shadow_reference_rgb_l1_255']>1.2*b['shadow_reference_rgb_l1_255']+1:reasons.append(co+': shadow colour')
        rankings.append({'id':name,'rejection_reasons':reasons,
            'background':max(metrics[c][name]['summary']['background_changed_fraction'] for c in cohorts),
            'shadow_rgb':max(metrics[c][name]['summary']['shadow_reference_rgb_l1_255'] for c in cohorts),
            'boundary_gradient':max(audits[c][name]['target_boundary_gradient_l1_255'] for c in cohorts),
            'cpu_seconds':max(metrics[c][name]['summary']['cpu_seconds'] for c in cohorts)})
    eligible=[r for r in rankings if not r['rejection_reasons']]
    if not eligible:raise ValueError('No candidate met frozen minimums; keep failure and do not open test')
    best=min(r['background'] for r in eligible);band=max(.0005,.05*best)
    chosen=min([r for r in eligible if r['background']<=best+band],key=lambda r:(r['shadow_rgb'],r['boundary_gradient'],r['cpu_seconds']))
    config=next(r['config'] for r in plan if r['id']==chosen['id'])
    final=[{'id':'prior_selected','engine':'research','config':json.load(open('configs/selected.json'))},
           {'id':'conservative','engine':'research','config':json.load(open('configs/research_conservative_known5.json'))},
           {'id':'selected','engine':'research','config':config}]
    outputs[1].write_text(json.dumps(config,indent=2)+'\n');outputs[2].write_text(json.dumps(final,indent=2)+'\n')
    payload={'frozen_before_test':True,'selected_id':chosen['id'],'selected_config':config,'rankings':rankings,'background_band':band,
             'plan_sha256':sha256(outputs[2]),'rules_sha256':sha256(public/'selection_rules.json'),
             'validation_metrics_sha256':{c:sha256(root/f'val_{c}'/'metrics.json') for c in cohorts},
             'holdout_manifest_sha256':{c:sha256(f'data/background_first_holdout20/{c}_final_test.jsonl') for c in cohorts}}
    outputs[0].write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();select(a.campaign)
