import argparse
import json
from .core import sample
from .targets import TARGETS
from .demo import demo
from .quadratic import quadratic_map


def main():
    p=argparse.ArgumentParser(description='Auditable fixed-setting two-dimensional HMC')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    q=sub.add_parser('quadratic');q.add_argument('input')
    q=sub.add_parser('sample');q.add_argument('input')
    a=p.parse_args()
    try:
        if a.command=='demo':result=demo()
        else:
            with open(a.input,encoding='utf8') as f:v=json.load(f)
            if a.command=='quadratic':
                if not isinstance(v,dict) or set(v)!={'precision','mass','step_size','steps','position'}:raise ValueError('quadratic input requires precision/mass/step_size/steps/position')
                print(json.dumps(quadratic_map(**v),indent=2,allow_nan=False));return
            required={'target','initial','method','draws','warmup','step_size','steps','mass','seed'}
            if not isinstance(v,dict) or set(v)!=required:raise ValueError('input requires exact target/initial/method/draws/warmup/step_size/steps/mass/seed fields')
            name=v.pop('target')
            if name not in TARGETS:raise ValueError('unknown target')
            result=sample(TARGETS[name],**v)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,f'hamiltonlab: {e}\n')
