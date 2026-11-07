from .core import sample,leapfrog
from .targets import correlated,mixture,TRUTH
from .diagnostics import moments,classic_rhat


def demo():
    arms=[]
    starts=[[-2.,-2.],[2.,2.],[-1.,-1.],[1.,1.]]
    for method,eps,steps in [('hmc',.15,10),('random_walk',.15,1)]:
        chains=[sample(correlated,q,method,1000,200,eps,steps,seed=71+i) for i,q in enumerate(starts)]
        pooled=[x for c in chains for x in c['samples']];summary=moments(pooled);summary.pop('x_sign_transitions')
        arms.append({'target':'correlated','chains':chains,'summary':summary,'rhat':classic_rhat([c['samples'] for c in chains]),'truth':TRUTH['correlated']})
    trapped=[sample(mixture,q,'hmc',1000,200,.15,10,seed=71+i) for i,q in enumerate([[-6.,0.],[-6.,0.],[6.,0.],[6.,0.]])]
    trap={'target':'mixture','chains':trapped,'summary':moments([x for c in trapped for x in c['samples']]),'chain_summaries':[moments(c['samples']) for c in trapped],
          'rhat':classic_rhat([c['samples'] for c in trapped]),'truth':TRUTH['mixture']}
    trap['summary'].pop('x_sign_transitions')
    assert all(s['x_sign_transitions']==0 for s in trap['chain_summaries'])
    assert all(c['retained_acceptance_rate']>.9 for c in trapped)
    assert trap['rhat']['coordinates'][0]['value']>3
    assert max(abs(x) for x in arms[0]['summary']['mean'])<.15
    q=[.4,-.2];p=[.3,-.6];forward=leapfrog(correlated,q,p,.1,13)
    back=leapfrog(correlated,forward['position'],[-z for z in forward['momentum']],.1,13)
    reverse=max(abs(back['position'][i]-q[i]) for i in range(2))
    assert reverse<1e-12
    return {'arms':arms,'separated_mode_failure':trap,'reversibility_position_max_abs_error':reverse,
            'development_protocol':'fixed seeds71..74,1000 retained +200 discarded per chain; settings chosen for a readable development illustration, not untouched test or parameter sweep; equal transitions, unequal target-call work; no speed ranking',
            'limits':'high acceptance and pooled moments do not prove exploration; all finite-chain results are descriptive, not certified posterior inference'}
