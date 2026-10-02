#!/usr/bin/env python3
"""Compute accuracy, precision/recall/F1 and confusion matrices for DR labels."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def norm(x): return str(x or '').strip().casefold().replace('_',' ')
def rows(path, dataset):
    out=[]
    for line in Path(path).open(encoding='utf-8'):
        if not line.strip(): continue
        r=json.loads(line)
        if dataset and str(r.get('dataset','')).casefold()!=dataset.casefold():
            # method outputs use dataset; aggregate ToG does too
            continue
        gold=(r.get('gold') or {}).get('label') or r.get('gold_label')
        o=r.get('output') or {}
        preds=o.get('labels') or r.get('generated_labels') or []
        pred=preds[0] if isinstance(preds,list) and preds else (r.get('final') or {}).get('pred_label')
        out.append((norm(gold),norm(pred) or 'null',r.get('sample_id')))
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='DR'); p.add_argument('--input',action='append',required=True,help='NAME=JSONL'); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    report={'dataset':a.dataset,'methods':{}}
    for spec in a.input:
        name,path=spec.split('=',1); rs=rows(path,a.dataset); gold_labels=sorted({g for g,_,_ in rs}); labels=sorted(set(gold_labels)|{q for _,q,_ in rs})
        cm=Counter((g,q) for g,q,_ in rs); n=len(rs); correct=sum(cm[(x,x)] for x in gold_labels)
        per={}
        for lab in gold_labels:
            tp=cm[(lab,lab)]; fp=sum(cm[(g,lab)] for g in labels if g!=lab); fn=sum(cm[(lab,q)] for q in labels if q!=lab)
            pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.; f=2*pr*re/(pr+re) if pr+re else 0.
            per[lab]={'support':sum(cm[(lab,q)] for q in labels),'precision':pr,'recall':re,'f1':f}
        macro=lambda k: sum(v[k] for v in per.values())/len(per) if per else 0.
        report['methods'][name]={'n':n,'accuracy':correct/n if n else 0.,'correct':correct,'wrong':n-correct,'macro_precision':macro('precision'),'macro_recall':macro('recall'),'macro_f1':macro('f1'),'per_label':per,'confusion_matrix':{g:{q:cm[(g,q)] for q in labels} for g in gold_labels},'sample_ids':[sid for _,_,sid in rs]}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:{m:x[k] for k in ('n','accuracy','macro_f1')} for m,x in report['methods'].items() for k in []},ensure_ascii=False));
    for m,x in report['methods'].items(): print(f'{m}: n={x["n"]} accuracy={x["accuracy"]:.4f} macro_f1={x["macro_f1"]:.4f}')
if __name__=='__main__': main()
