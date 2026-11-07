"""Pooled average ranks, split chains and folded rank-normalized R-hat."""
import math
from statistics import NormalDist,median
from .core import vector,real


def normal_scores(values):
    if not isinstance(values,list) or not 2<=len(values)<=100000:raise ValueError('normal_scores needs2..100000 values')
    x=[real(v,'rank input') for v in values];n=len(x);order=sorted(range(n),key=x.__getitem__);ranks=[0.]*n
    start=0
    while start<n:
        end=start+1
        while end<n and x[order[end]]==x[order[start]]:end+=1
        rank=(start+1+end)/2
        for i in order[start:end]:ranks[i]=rank
        start=end
    scores=[NormalDist().inv_cdf((r-.375)/(n+.25)) for r in ranks]
    return {'average_ranks':ranks,'normal_scores':scores,'total_draws':n}


def scalar_rhat(chains):
    n=len(chains[0]);m=len(chains);means=[math.fsum(c)/n for c in chains];grand=math.fsum(means)/m
    within=math.fsum(math.fsum((z-mu)**2 for z in c)/(n-1) for c,mu in zip(chains,means))/m
    between=n*math.fsum((mu-grand)**2 for mu in means)/(m-1)
    if not within:return {'value':None,'status':'undefined_zero_within_variance','within_chain_variance':within,'between_chain_variance':between}
    value=math.sqrt(((n-1)*within/n+between/n)/within)
    return {'value':value,'status':'computed','within_chain_variance':within,'between_chain_variance':between}


def rank_split_rhat(chains,include_audit=False):
    if type(include_audit) is not bool:raise ValueError('include_audit must be bool')
    if not isinstance(chains,list) or not 2<=len(chains)<=64:raise ValueError('need2..64 chains')
    if not all(isinstance(c,list) for c in chains):raise ValueError('chains must be lists')
    n=len(chains[0]);m=len(chains)
    if n<4 or n%2 or any(len(c)!=n for c in chains):raise ValueError('equal even chain lengths>=4 required; no silent draw dropping')
    if n*m>100000:raise ValueError('total draws exceeds100000')
    x=[[vector(p,'sample') for p in c] for c in chains];half=n//2
    split=[part for c in x for part in [c[:half],c[half:]]];out=[]
    for k in range(2):
        raw=[p[k] for c in split for p in c];center=median(raw);fold=[abs(z-center) for z in raw]
        if not all(math.isfinite(z) for z in fold):raise ValueError('nonfinite folded difference')
        rank=normal_scores(raw);foldrank=normal_scores(fold)
        def component(v):return scalar_rhat([v[i:i+half] for i in range(0,len(v),half)])
        bulk=component(rank['normal_scores']);tail=component(foldrank['normal_scores'])
        valid=bulk['value'] is not None and tail['value'] is not None
        entry={'rank_normalized_split':bulk,'folded_rank_normalized_split':tail,
               'max_rhat':max(bulk['value'],tail['value']) if valid else None,
               'status':'computed' if valid else 'undefined_component','pooled_median':center}
        if include_audit:entry['audit']={'raw_split_order':raw,'folded_split_order':fold,'rank':rank,'folded_rank':foldrank}
        out.append(entry)
    return {'coordinates':out,'original_chains':m,'original_draws_per_chain':n,'split_chains':2*m,'split_draws_per_chain':half,
            'retained_draws':n*m,'dropped_draws':0,
            'semantics':'maximum of pooled rank-normalized split and folded split R-hat; average tied ranks, Blom offset3/8; equal even lengths only; no ESS, MCSE or convergence certificate; undefined components remain null'}
