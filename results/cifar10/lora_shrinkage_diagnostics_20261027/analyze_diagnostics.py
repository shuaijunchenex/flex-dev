"""Audit and summarize the actual controlled-inference outputs, never training."""
from pathlib import Path
from collections import defaultdict
import json,csv,math,statistics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostics'
ROUNDS={9,19,29,39,49,59,69,79,89,100}
TREATMENTS={'rbla','support_zero','norm_matched','permuted_reverse','permuted_halfshift'}
status=json.loads((OUT/'status.json').read_text())
rows=list(csv.DictReader((OUT/'evaluations.csv').open())) if (OUT/'evaluations.csv').exists() else []
layers=list(csv.DictReader((OUT/'layer_metrics.csv').open())) if (OUT/'layer_metrics.csv').exists() else []
groups=defaultdict(dict)
invalid=[]
for row in rows:
    if row.get('status')!='ok' or int(row.get('samples') or 0)!=10000:
        invalid.append(row);continue
    if not all(math.isfinite(float(row[k])) for k in ('accuracy','average_loss','paired_logit_rmse')):
        invalid.append(row);continue
    key=(int(row['round']),row['scope'],row['budget'])
    assert row['treatment'] not in groups[key],f'Duplicate evaluation: {key}'
    groups[key][row['treatment']]=row
contrasts=[]
for (round_,scope,budget),t in sorted(groups.items()):
    if set(t)!=TREATMENTS:continue
    base=t['rbla'];zero=t['support_zero'];norm=t['norm_matched']
    contrasts.append({'round':round_,'scope':scope,'budget':budget,
        'rbla_acc':float(base['accuracy']),'support_zero_acc':float(zero['accuracy']),
        'norm_matched_acc':float(norm['accuracy']),
        'reverse_acc':float(t['permuted_reverse']['accuracy']),
        'halfshift_acc':float(t['permuted_halfshift']['accuracy']),
        'zero_minus_rbla_pp':100*(float(zero['accuracy'])-float(base['accuracy'])),
        'zero_minus_rbla_loss':float(zero['average_loss'])-float(base['average_loss']),
        'zero_minus_norm_pp':100*(float(zero['accuracy'])-float(norm['accuracy'])),
        'zero_minus_norm_loss':float(zero['average_loss'])-float(norm['average_loss']),
        'zero_minus_reverse_pp':100*(float(zero['accuracy'])-float(t['permuted_reverse']['accuracy'])),
        'zero_minus_halfshift_pp':100*(float(zero['accuracy'])-float(t['permuted_halfshift']['accuracy'])),
        'zero_flip_rate':float(zero['prediction_flip_rate']),
        'zero_logit_rmse':float(zero['paired_logit_rmse'])})
agg=[]
for scope in ('all_conv','first_conv','head_only'):
    for budget in ('max','mid','min'):
        subset=[c for c in contrasts if c['scope']==scope and c['budget']==budget]
        if not subset:continue
        metrics=('zero_minus_rbla_pp','zero_minus_rbla_loss','zero_minus_norm_pp','zero_minus_norm_loss','zero_minus_reverse_pp','zero_minus_halfshift_pp')
        record={'scope':scope,'budget':budget,'checkpoint_count':len(subset)}
        for key in metrics:record['mean_'+key]=statistics.mean(c[key] for c in subset)
        record['zero_accuracy_improvements']=sum(c['zero_minus_rbla_pp']>1e-10 for c in subset)
        record['zero_loss_improvements']=sum(c['zero_minus_rbla_loss'] < -1e-10 for c in subset)
        agg.append(record)
min_rows=[c for c in contrasts if c['budget']=='min']
min_check={'comparison_count':len(min_rows),
    'max_abs_accuracy_delta_pp':max((abs(c['zero_minus_rbla_pp']) for c in min_rows),default=None),
    'max_abs_loss_delta':max((abs(c['zero_minus_rbla_loss']) for c in min_rows),default=None),
    'max_logit_rmse':max((c['zero_logit_rmse'] for c in min_rows),default=None),
    'max_flip_rate':max((c['zero_flip_rate'] for c in min_rows),default=None)}
probe_rows=[r for r in layers if r['metric_type']=='same_input_probe']
assert all(int(r['samples'])==128 for r in probe_rows)
present={int(r['round']) for r in rows if r.get('status')=='ok'}
reports=[]
for p in sorted((OUT/'checkpoints').glob('*.json')):
    report=json.loads(p.read_text());reports.append({'file':str(p),'round':report['round'],'status':report['metadata']['status'],'samples':report['metadata']['dataset_samples'],'probe_samples':report['metadata']['probe_samples']})
full=(status['state']=='completed' and not status.get('failed') and not status.get('invalid_treatments') and not invalid and present==ROUNDS and len(contrasts)==90 and len(rows)==450 and len(reports)==10 and all(r['status']=='complete' and r['samples']==10000 and r['probe_samples']==128 for r in reports))
analysis={'acceptance':'complete' if full else 'incomplete_or_in_progress','source_status':status,'valid_evaluation_rows':len(rows)-len(invalid),'invalid_rows':invalid,'missing_rounds':sorted(ROUNDS-present),'complete_contrasts':len(contrasts),'checkpoint_reports':reports,'minimum_budget_check':min_check,'scope_budget_summary':agg,'contrasts':contrasts,'interpretation_limits':['All interventions share one snapshot and uploads; support_zero is not an independently trained ZeroPadding model.','Checkpoint averages are descriptive correlated observations, not independent-seed significance tests.','Tail slot indices do not establish high-frequency features.','Immediate inference effects do not prove long-term regularization or client bias.']}
(ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
for filename,data in [('paired_contrasts.csv',contrasts),('scope_budget_summary.csv',agg)]:
    if data:
        with (ROOT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
lines=['CIFAR-10 controlled snapshot diagnostic audit',f'Acceptance: {analysis["acceptance"]}',f'Missing rounds: {analysis["missing_rounds"]}',f'Valid evaluation rows: {analysis["valid_evaluation_rows"]}/450',f'Complete contrasts: {len(contrasts)}/90',f'Minimum budget: {min_check}','']
lines.extend(str(r) for r in agg)
(ROOT/'analysis.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
