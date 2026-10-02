#!/usr/bin/env python3
"""Extract gloss-enriched claims from an aggregated explanation JSONL."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from config.settings import Settings
from evaluation.local_graph_eval import LocalGraphEvaluator

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); rows=[json.loads(x) for x in a.input.read_text(encoding='utf-8').splitlines() if x.strip()]
    e=LocalGraphEvaluator(); s=Settings(); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8') as out:
      for i,r in enumerate(rows,1):
       try:
        text=r.get('generated_explanation','').strip()
        if not text: raise ValueError('empty generated_explanation')
        entities,claims,usage=e.extract(text,'T')
        result={**r,'entities':entities,'claims':[c.as_dict() for c in claims],'usage':usage,'extractor':{'model':s.llm_model,'api_base':s.llm_api_base},'status':'ok'}
       except Exception as err:
        result={**r,'entities':{'case':[],'clinical':[]},'claims':[],'status':'error','error':f'{type(err).__name__}: {err}'}
       out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush(); print(f'[{i}/{len(rows)}] {r.get("setup")} {r.get("sample_id")} {result["status"]}',flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
