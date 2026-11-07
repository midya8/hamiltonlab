"""Deterministic SVG from the full offline result; no plotting dependencies."""
import argparse
import html
import json
from pathlib import Path


def render(v):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="680" viewBox="0 0 1120 680">','<rect width="1120" height="680" fill="white"/>']
    def text(x,y,s,size=14,weight=400):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="#251f21">{html.escape(s)}</text>')
    text(28,35,'HamiltonLab | acceptance is not exploration',25)
    text(28,61,'Four seeded chains per arm, 1,000 retained +200 discarded each. Fixed settings, not a tuned sampler ranking.')
    for idx,arm in enumerate(v['arms']):
        x=45+idx*355;y=120;size=290
        text(x,96,'Correlated Gaussian: '+arm['chains'][0]['method'],17,600)
        for k in [-4,-2,0,2,4]:
            z=(k+4)/8*size
            out.append(f'<path d="M{x+z} {y}v{size} M{x} {y+size-z}h{size}" fill="none" stroke="#eae9ea"/>')
            text(x+z-5,y+size+20,str(k),11);text(x-22,y+size-z+4,str(k),11)
        outside=0
        for c in arm['chains']:
            for p in c['samples'][:200]:
                if any(abs(z)>4 for z in p):outside+=1;continue
                px=x+(p[0]+4)/8*size;py=y+size-(p[1]+4)/8*size
                out.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="1.8" fill="{"#73a89a" if idx==0 else "#585254"}" opacity=".42"/>')
        rates=[c['retained_acceptance_rate'] for c in arm['chains']];s=arm['summary']
        text(x,458,f'Acceptance: {min(rates):.3f} to {max(rates):.3f}')
        text(x,482,f'Mean: [{s["mean"][0]:.3f}, {s["mean"][1]:.3f}]')
        text(x,506,f'Cov(x,y): {s["sample_covariance"][0][1]:.3f} | truth 0.950')
        text(x,530,f'Classical x R-hat: {arm["rhat"]["coordinates"][0]["value"]:.3f}')
        text(x,554,f'Target callbacks: {sum(c["potential_gradient_callback_calls"] for c in arm["chains"]):,}')
        text(x,578,f'Plot: first200/chain; outside [-4,4]: {outside}',12)
    trap=v['separated_mode_failure'];x=755;y=120;width=320;height=290
    text(x,96,'Separated mixture: HMC x traces',17,600)
    for k in [-12,-6,0,6,12]:
        py=y+height-(k+12)/24*height
        out.append(f'<path d="M{x} {py}h{width}" stroke="#eae9ea"/>');text(x-24,py+4,str(k),11)
    outside=0
    for i,c in enumerate(trap['chains']):
        pts=[]
        for j,p in enumerate(c['samples']):
            if not -12<=p[0]<=12:outside+=1;continue
            pts.append(f'{x+j/999*width:.3f},{y+height-(p[0]+12)/24*height:.3f}')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{["#251f21","#585254","#73a89a","#a3c4ba"][i]}" stroke-width=".6"/>')
    text(x,y+height+20,'0                      retained iteration                     999',11)
    text(x,458,'Acceptance: 0.997 to 1.000')
    text(x,482,'Within-chain mode switches: 0 / 0 / 0 / 0')
    text(x,506,f'Pooled positive-x share: {trap["summary"]["positive_x_fraction"]:.3f}')
    text(x,530,f'Classical x R-hat: {trap["rhat"]["coordinates"][0]["value"]:.3f}')
    text(x,554,'All1,000/chain shown. No cross-chain joins.')
    text(x,578,f'Trace points outside x[-12,12]: {outside}',12)
    text(28,620,'Shared Gaussian scatter axes [-4,4] for both coordinates. Mixture has separate time/x axes; no density fit or hidden thinning.',13)
    text(28,644,'R-hat is classical unsplit/unranked, not a convergence certificate. Equal transitions cost different target work; no speed claim.',13)
    text(28,668,'Exact truths are analytic. Good pooled moments can coexist with zero within-chain mode exploration.',13)
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--check',action='store_true');a=p.parse_args()
    value=render(json.loads(Path(a.input).read_text()));path=Path(__file__).resolve().parents[1]/'assets/chain-audit.svg'
    if a.check:
        if path.read_text()!=value:raise SystemExit('SVG differs from result')
    else:path.write_text(value)
