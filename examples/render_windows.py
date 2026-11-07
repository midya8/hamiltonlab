"""Reproducible finite-window figure and full curve-data JSON, standard library."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hamiltonlab.finite_window import finite_window_audit,mixture_window_audit


def curves():
    specs=[('quarter','Quarter period','Fixed T = pi/2',[math.pi/2]),
           ('flip','Sign flip','Fixed T = pi',[math.pi]),
           ('identity','Identity','Fixed T = 2pi',[2*math.pi]),
           ('random_flip','Random sign / identity','IID equal choice: pi or 2pi',[math.pi,2*math.pi]),
           ('random_nonresonant','Random nonresonant','IID equal choice: pi/3 or 2pi/3',[math.pi/3,2*math.pi/3])]
    arms=[]
    for key,title,subtitle,times in specs:
        trajectory=[{'step_size':t/8,'steps':8} for t in times];points=[]
        for n in range(1,41):
            r=(finite_window_audit([1.,1.],[1.,1.],trajectory[0]['step_size'],8,n) if len(times)==1 else mixture_window_audit([1.,1.],[1.,1.],trajectory,n))
            c=r['coordinates'][0]
            points.append({'draws':n,'position_factor':c['position']['variance_of_mean_over_stationary_variance'],
                           'squared_position_factor':c['squared_position']['variance_of_mean_over_stationary_variance']})
        arms.append({'key':key,'title':title,'subtitle':subtitle,'trajectories':trajectory,'points':points})
    return {'precision':[1.,1.],'mass':[1.,1.],'arms':arms,
            'semantics':'hypothetical stationary exact-flow Gaussian kernels, not recorded leapfrog chains; vertical value is variance of N-draw observable mean divided by stationary variance of that SAME observable; IID reference1/N; finite N1..40 with every integer retained; no ESS/MCSE or convergence claim'}


def validate(v):
    if not isinstance(v,dict) or v.get('precision')!=[1.,1.] or v.get('mass')!=[1.,1.]:raise ValueError('figure requires unit precision/mass')
    if not isinstance(v.get('arms'),list) or len(v['arms'])!=5:raise ValueError('figure requires five arms')
    expected=['quarter','flip','identity','random_flip','random_nonresonant']
    for arm,key in zip(v['arms'],expected):
        if arm.get('key')!=key or not isinstance(arm.get('title'),str) or not isinstance(arm.get('subtitle'),str):raise ValueError('invalid arm identity or label')
        points=arm.get('points')
        if not isinstance(points,list) or len(points)!=40:raise ValueError('requires40 integer window lengths')
        for n,p in enumerate(points,1):
            if type(p.get('draws')) is not int or p['draws']!=n:raise ValueError('draws must be ordered1..40')
            for name in ['position_factor','squared_position_factor']:
                z=p.get(name)
                if isinstance(z,bool) or not isinstance(z,(int,float)) or not math.isfinite(z) or not 0<=z<=1.000000000000001:raise ValueError('invalid normalized variance')


def render(v):
    validate(v)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1080" viewBox="0 0 1120 1080" role="img" aria-labelledby="title desc">',
         '<title id="title">HamiltonLab finite-window observable variance</title>',
         '<desc id="desc">Five exact-flow Gaussian references compare position and squared-position averages, including zero position correlation with frozen magnitude. Same linear axes in every panel, N1 through40, variance factor0 through1.</desc>',
         '<rect width="1120" height="1080" fill="white"/>']
    def text(x,y,s,size=16,weight=400):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#251f21">{html.escape(s)}</text>')
    text(30,38,'HamiltonLab | a stable mean can hide frozen magnitude',27)
    text(30,68,'Exact Gaussian flow references, stationary start. Not sampled leapfrog trajectories or convergence evidence.',16)
    for x,color,label,dash in [(30,'#73a89a','q mean',''),(260,'#251f21','q-squared mean','6 3'),(570,'#999396','IID reference1/N','2 5')]:
        out.append(f'<path d="M{x} 95h45" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>');text(x+55,101,label,17,600)
    for idx,arm in enumerate(v['arms']):
        x=75+(idx%2)*550;y=165+(idx//2)*285;w=430;h=160
        text(x,y-35,arm['title'],20,600);text(x,y-12,arm['subtitle'],16)
        for z in [0,.25,.5,.75,1.]:
            yy=y+h-z*h;out.append(f'<path d="M{x} {yy}h{w}" stroke="#eae9ea"/>');text(x-40,yy+5,f'{z:g}',14)
        for n in [1,10,20,30,40]:
            xx=x+(n-1)/39*w;out.append(f'<path d="M{xx} {y}v{h}" stroke="#eae9ea"/>');text(xx-8,y+h+23,str(n),14)
        text(x+140,y+h+47,'Window length N',16)
        for field,color,dash in [('position_factor','#73a89a',''),('squared_position_factor','#251f21','6 3'),('iid','#999396','2 5')]:
            pts=[]
            for p in arm['points']:
                value=1/p['draws'] if field=='iid' else p[field]
                pts.append(f'{x+(p["draws"]-1)/39*w:.6f},{y+h-value*h:.6f}')
            out.append(f'<polyline data-arm="{arm["key"]}" data-observable="{field}" points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.4" stroke-dasharray="{dash}"/>')
    text(625,740,'Read the vertical axis as',20,600)
    text(625,771,'Var(observable mean) / Var(observable)',17)
    text(625,810,'Shared linear scale0..1. Every integer N1..40.',16)
    text(625,841,'At N20, random sign / identity:',17,600)
    text(625,869,'q factor = .05, q-squared factor = 1.',17)
    text(625,899,'Position correlations vanish; magnitude stays fixed.',16)
    text(30,1030,'Lines join integer window-length summaries, not sample paths. Zero variance of one average is not exploration.',16)
    text(30,1058,'Full curve data: python examples/render_windows.py --json /tmp/window-curves.json | see README derivations and limits.',15)
    out.append('</svg>');return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--json');a=p.parse_args()
    v=curves();svg=render(v);path=Path(__file__).resolve().parents[1]/'assets/window-audit.svg'
    if a.json:Path(a.json).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    if a.check:
        if path.read_text()!=svg:raise SystemExit('window SVG differs from source curves')
    else:path.write_text(svg)

if __name__=='__main__':main()
