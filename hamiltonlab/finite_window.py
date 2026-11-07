"""Finite-window variances for a hypothetical stationary exact-flow kernel."""
import math
from .core import integer
from .harmonic import harmonic_audit


def _window(rho,n):
    """Nonnegative innovation form avoids cancellation at rho=-1, even n.

    a_j=1+rho+...+rho^(j-1). Var(sum Z)=a_n²+(1-rho²)*sum_{j<n}a_j²
    for standardized stationary AR(1) covariance rho^|i-j|. This formula
    depends only on covariance, not on Gaussian innovations for q².
    """
    correlations=[1.];r=1.;a=0.;squares=[]
    for j in range(1,n+1):
        a=1+rho*a
        if j<n:
            squares.append(a*a);r*=rho;correlations.append(r)
    innovation_variance=(1-rho)*(1+rho)
    sum_variance=math.fsum([a*a,innovation_variance*math.fsum(squares)])
    factor=sum_variance/(n*n)
    weighted=[(1-lag/n)*correlations[lag] for lag in range(1,n)]
    signed_lag_factor=(1+2*math.fsum(weighted))/n
    return {'lag_correlations':correlations,'lag_weights':[1-lag/n for lag in range(n)],
            'weighted_lag_terms':weighted,'initial_state_sum_coefficient':a,
            'innovation_variance':innovation_variance,'innovation_coefficient_squares_sum':math.fsum(squares),
            'variance_of_mean_over_stationary_variance':factor,
            'signed_lag_formula_factor':signed_lag_factor,
            'signed_lag_formula_minus_nonnegative_factor':signed_lag_factor-factor,
            'variance_ratio_to_iid_mean':factor*n}


def finite_window_audit(precision,mass,step_size,steps,draws):
    n=integer(draws,1,10000,'draws')
    flow=harmonic_audit(precision,mass,step_size,steps);coords=[]
    for k,c in zip(flow['leapfrog']['precision'],flow['coordinates']):
        rho=c['exact_stationary_lag_one_position_correlation']
        position=_window(rho,n);square=_window(rho*rho,n)
        for audit,mean,variance in [(position,0.,1/k),(square,1/k,2/k/k)]:
            audit.update({'stationary_mean':mean,'stationary_variance':variance,
                          'iid_variance_of_mean':variance/n,
                          'variance_of_mean':variance*audit['variance_of_mean_over_stationary_variance']})
            if not all(math.isfinite(audit[name]) for name in ['stationary_mean','stationary_variance','iid_variance_of_mean','variance_of_mean']):raise ValueError('nonfinite finite-window variance')
        coords.append({'frequency':c['frequency'],'phase':c['phase'],'position':position,'squared_position':square})
    return {'draws':n,'duration':flow['duration'],'precision':flow['leapfrog']['precision'],'mass':flow['leapfrog']['mass'],'coordinates':coords,
            'semantics':'analytic finite-window covariance for N consecutive positions of a hypothetical stationary exact-flow full-momentum-refresh diagonal Gaussian kernel; stationary initial draw required; squared-position covariance uses Gaussian fourth moments; not an ESS, CLT, MCSE estimate or diagnostic of the actual Metropolized leapfrog samples; roundoff is retained in the signed lag audit, nonnegative innovation form supplies variance'}


def mixture_window_audit(precision,mass,trajectories,draws):
    """IID equal-probability duration choices, independent of state/momentum.

    The same duration is selected for both coordinates. Cross-coordinate
    higher moments need not factorize under this shared selection; only
    marginal q and q² observable covariances are reported here.
    """
    n=integer(draws,1,10000,'draws')
    if not isinstance(trajectories,list) or not 1<=len(trajectories)<=64:raise ValueError('trajectories must contain1..64 entries')
    flows=[]
    for trajectory in trajectories:
        if not isinstance(trajectory,dict) or set(trajectory)!={'step_size','steps'}:raise ValueError('trajectory requires exact step_size/steps fields')
        flows.append(harmonic_audit(precision,mass,**trajectory))
    coords=[];kvec=flows[0]['leapfrog']['precision'];mvec=flows[0]['leapfrog']['mass']
    for i,k in enumerate(kvec):
        branches=[{'phase':flow['coordinates'][i]['phase'],
                   'duration':flow['duration'],'probability':1/len(flows),
                   'cosine':flow['coordinates'][i]['exact_stationary_lag_one_position_correlation']} for flow in flows]
        a=math.fsum(b['cosine']/len(branches) for b in branches)
        b=math.fsum(b['cosine']**2/len(branches) for b in branches)
        position=_window(a,n);square=_window(b,n)
        for audit,mean,variance in [(position,0.,1/k),(square,1/k,2/k/k)]:
            audit.update({'stationary_mean':mean,'stationary_variance':variance,
                          'iid_variance_of_mean':variance/n,
                          'variance_of_mean':variance*audit['variance_of_mean_over_stationary_variance']})
            if not all(math.isfinite(audit[name]) for name in ['stationary_mean','stationary_variance','iid_variance_of_mean','variance_of_mean']):raise ValueError('nonfinite mixture-window variance')
        coords.append({'branches':branches,'mean_cosine':a,'mean_squared_cosine':b,
                       'square_of_mean_cosine':a*a,'cosine_variance':math.fsum((r['cosine']-a)**2/len(branches) for r in branches),
                       'position':position,'squared_position':square})
    return {'draws':n,'precision':kvec,'mass':mvec,'trajectories':[dict(t) for t in trajectories],
            'selection':'each listed entry is equally likely, IID across transitions and independent of current state and fresh momentum; duplicates retain multiplicity; one shared branch per transition for both coordinates',
            'coordinates':coords,
            'semantics':'hypothetical stationary exact-flow Gaussian mixture kernel, not the sampled leapfrog random-length kernel; marginal observable covariance only, no joint-coordinate process independence claim, ESS/MCSE/CLT estimate or general mixing certificate; ordinary floating-point trigonometry at resonances'}
