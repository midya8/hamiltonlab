"""Descriptive finite-chain checks, deliberately not a modern convergence suite."""
import math
from .core import vector


def moments(samples):
    if not isinstance(samples,list) or len(samples)<2:raise ValueError('need at least two samples')
    x=[vector(z,'sample') for z in samples];n=len(x);mu=[math.fsum(p[k] for p in x)/n for k in range(2)]
    cov=[[math.fsum((p[i]-mu[i])*(p[j]-mu[j]) for p in x)/(n-1) for j in range(2)] for i in range(2)]
    return {'mean':mu,'sample_covariance':cov,'positive_x_fraction':sum(p[0]>0 for p in x)/n,
            'x_sign_transitions':sum((x[i][0]>0)!=(x[i-1][0]>0) for i in range(1,n))}


def classic_rhat(chains):
    if not isinstance(chains,list) or len(chains)<2:raise ValueError('need two or more chains')
    n=len(chains[0]);m=len(chains)
    if n<2 or any(len(c)!=n for c in chains):raise ValueError('chains need equal length>=2')
    summaries=[moments(c) for c in chains];result=[]
    for k in range(2):
        means=[s['mean'][k] for s in summaries];grand=math.fsum(means)/m
        between=n*math.fsum((x-grand)**2 for x in means)/(m-1)
        within=math.fsum(s['sample_covariance'][k][k] for s in summaries)/m
        value=math.sqrt(((n-1)*within/n+between/n)/within) if within else None
        result.append({'value':value,'within_chain_variance':within,'between_chain_variance':between,
                       'status':'computed' if within else 'undefined_zero_within_variance'})
    return {'coordinates':result,'semantics':'classical unsplit unranked R-hat; no truncation at1, no ESS, MCSE or convergence certificate'}
