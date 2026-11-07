import argparse
import json
from .core import sample
from .targets import TARGETS
from .demo import demo
from .quadratic import quadratic_map
from .rankdiag import rank_split_rhat
from .harmonic import harmonic_audit
from .finite_window import finite_window_audit


def main():
    p=argparse.ArgumentParser(description='Auditable fixed-setting two-dimensional HMC')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    q=sub.add_parser('diagnose');q.add_argument('input')
    q=sub.add_parser('quadratic');q.add_argument('input')
    q=sub.add_parser('harmonic');q.add_argument('input')
    q=sub.add_parser('window');q.add_argument('input')
    q=sub.add_parser('sample');q.add_argument('input')
    a=p.parse_args()
    try:
        if a.command=='demo':result=demo()
        else:
            with open(a.input,encoding='utf8') as f:v=json.load(f)
            if a.command=='window':
                if not isinstance(v,dict) or set(v)!={'precision','mass','step_size','steps','draws'}:raise ValueError('window requires exact precision/mass/step_size/steps/draws fields')
                print(json.dumps(finite_window_audit(**v),indent=2,allow_nan=False));return
            if a.command=='diagnose':
                if not isinstance(v,dict) or set(v)!={'chains'}:raise ValueError('diagnose input requires exact chains field')
                print(json.dumps(rank_split_rhat(v['chains'],include_audit=True),indent=2,allow_nan=False));return
            if a.command in ('quadratic','harmonic'):
                if not isinstance(v,dict) or set(v)!={'precision','mass','step_size','steps','position'}:raise ValueError('quadratic input requires precision/mass/step_size/steps/position')
                print(json.dumps((harmonic_audit if a.command=='harmonic' else quadratic_map)(**v),indent=2,allow_nan=False));return
            required={'target','initial','method','draws','warmup','step_size','steps','mass','seed'}
            if not isinstance(v,dict) or not required<=set(v) or set(v)-required-{'steps_range'}:raise ValueError('input requires exact target/initial/method/draws/warmup/step_size/steps/mass/seed fields')
            name=v.pop('target')
            if name not in TARGETS:raise ValueError('unknown target')
            result=sample(TARGETS[name],**v)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,f'hamiltonlab: {e}\n')
