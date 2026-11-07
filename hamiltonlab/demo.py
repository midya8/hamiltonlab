from .core import sample,leapfrog
from .targets import correlated,mixture,standard,TRUTH
from .quadratic import quadratic_map
from .rankdiag import rank_split_rhat
import math
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
    resonance=[]
    for length in [1,4]:
        chains=[sample(standard,q,'hmc',200,0,math.sqrt(2),length,seed=71+i) for i,q in enumerate(starts)]
        audits=[{'max_position_span':max(max(p[k] for p in c['samples'])-min(p[k] for p in c['samples']) for k in range(2)),
                 'mean_squared_jump':sum(sum((t['state'][k]-t['before'][k])**2 for k in range(2)) for t in c['transitions'])/200} for c in chains]
        resonance.append({'steps':length,'map':quadratic_map([1.,1.],[1.,1.],math.sqrt(2),length), 'chains':chains,'chain_audits':audits})
    assert max(a['max_position_span'] for a in resonance[1]['chain_audits'])<1e-10
    assert min(c['retained_acceptance_rate'] for c in resonance[1]['chains'])>.99
    assert min(a['mean_squared_jump'] for a in resonance[0]['chain_audits'])>.1
    randomized=[sample(standard,q,'hmc',200,0,math.sqrt(2),4,seed=71+i,steps_range=[1,4]) for i,q in enumerate(starts)]
    randomized_audits=[{'max_position_span':max(max(p[k] for p in c['samples'])-min(p[k] for p in c['samples']) for k in range(2)),
                       'mean_squared_jump':sum(sum((t['state'][k]-t['before'][k])**2 for k in range(2)) for t in c['transitions'])/200} for c in randomized]
    assert min(a['mean_squared_jump'] for a in randomized_audits)>.1
    assert all(c['potential_gradient_callback_calls']==1+sum(t['chosen_steps']+1 for t in c['transitions']) for c in randomized)
    scale_base=[-.5,.5,-1.5,1.5]*25
    scale_chains=[[[z*scale,z] for z in scale_base] for scale in [1.,1.,10.,10.]]
    trend_chains=[[[z,z] for z in ([-2.,-1.]*25+[1.,2.]*25)] for _ in range(4)]
    rank_audit={'correlated_hmc':rank_split_rhat([c['samples'] for c in arms[0]['chains']]),
                'correlated_random_walk':rank_split_rhat([c['samples'] for c in arms[1]['chains']]),
                'separated_modes':rank_split_rhat([c['samples'] for c in trapped]),
                'deterministic_scale_difference':{'chains':scale_chains,'classical':classic_rhat(scale_chains),'rank_split':rank_split_rhat(scale_chains,True)},
                'deterministic_half_chain_shift':{'chains':trend_chains,'classical':classic_rhat(trend_chains),'rank_split':rank_split_rhat(trend_chains,True)}}
    assert rank_audit['deterministic_scale_difference']['rank_split']['coordinates'][0]['folded_rank_normalized_split']['value']>1.2
    assert rank_audit['deterministic_half_chain_shift']['rank_split']['coordinates'][0]['rank_normalized_split']['value']>1.2
    return {'rank_diagnostics':rank_audit,'randomized_length_audit':{'chains':randomized,'chain_audits':randomized_audits,'selection':'uniform integer1..4 independent of state; same seed labels/starts/200 transitions as resonance arms but different random draws/work, not a controlled causal comparison'},'quadratic_resonance':resonance,'arms':arms,'separated_mode_failure':trap,'reversibility_position_max_abs_error':reverse,
            'development_protocol':'fixed seeds71..74,1000 retained +200 discarded per chain; settings chosen for a readable development illustration, not untouched test or parameter sweep; equal transitions, unequal target-call work; no speed ranking',
            'limits':'high acceptance and pooled moments do not prove exploration; all finite-chain results are descriptive, not certified posterior inference'}
