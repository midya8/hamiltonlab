"""Analytic diagonal-Gaussian flow reference, not a sampling implementation."""
import math
from .quadratic import quadratic_map


def harmonic_audit(precision,mass,step_size,steps,position=None):
    """Compare unflipped leapfrog with exact flow at the SAME signed time.

    H=(k*q*q+p*p/m)/2, omega=sqrt(k/m). In standardized phase space
    (sqrt(k)*q,p/sqrt(m)) exact flow is an orthogonal rotation.
    All numerical proposal moments remain BEFORE Metropolis correction.
    """
    numerical=quadratic_map(precision,mass,step_size,steps,position)
    duration=numerical['step_size']*numerical['steps'];coords=[]
    for i,num in enumerate(numerical['coordinates']):
        k=numerical['precision'][i];m=numerical['mass'][i];q=numerical['position'][i]
        rootk=math.sqrt(k);rootm=math.sqrt(m);scale=rootk*rootm
        omega=rootk/rootm;phase=omega*duration
        if not math.isfinite(phase) or abs(phase)>1e6:raise ValueError('absolute harmonic phase must be<=1e6')
        c=math.cos(phase);s=math.sin(phase)
        exact=[[c,s/scale],[-scale*s,c]]
        rotation=[[c,s],[-s,c]]
        a,b=num['trajectory_map'][0];cc,d=num['trajectory_map'][1]
        normalized=[[a,b*scale],[cc/scale,d]]
        error=[[normalized[r][t]-rotation[r][t] for t in range(2)] for r in range(2)]
        # A standardized stationary input has identity covariance. AA^T
        # measures the pre-correction pushforward's joint covariance.
        covariance=[[math.fsum(normalized[r][j]*normalized[t][j] for j in range(2)) for t in range(2)] for r in range(2)]
        exact_covariance=[[math.fsum(rotation[r][j]*rotation[t][j] for j in range(2)) for t in range(2)] for r in range(2)]
        frobenius=math.sqrt(math.fsum(v*v for row in error for v in row))
        exact_variance=(c*c+s*s)/k
        esjd=((c-1)**2+s*s)/k
        values=[v for matrix in [exact,rotation,normalized,error,covariance,exact_covariance] for row in matrix for v in row]
        values += [frobenius,c*q,s*s/k,exact_variance,esjd,num['stationary_input_proposal_variance']*k-1]
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite harmonic audit')
        coords.append({'frequency':omega,'phase':phase,'exact_trajectory_map':exact,
                       'exact_standardized_rotation':rotation,'leapfrog_standardized_map':normalized,
                       'standardized_map_error':error,'standardized_map_error_frobenius':frobenius,
                       'exact_standardized_stationary_joint_covariance':exact_covariance,
                       'leapfrog_standardized_stationary_joint_covariance':covariance,
                       'exact_determinant':c*c+s*s,
                       'exact_conditional_position_mean':c*q,'exact_conditional_position_variance':s*s/k,
                       'exact_stationary_position_variance':exact_variance,
                       'exact_stationary_expected_squared_jump':esjd,
                       'exact_stationary_lag_one_position_correlation':c,
                       'leapfrog_stationary_position_variance_relative_error':num['stationary_input_proposal_variance']*k-1})
    return {'duration':duration,'coordinates':coords,'leapfrog':numerical,
            'semantics':'analytic separable Gaussian flow reference at the same signed duration as leapfrog; stationary position/momentum independent Gaussian inputs; correlation is for a hypothetical exact-flow fresh-momentum kernel, not the accepted numerical chain; pre-Metropolis leapfrog covariance is not its invariant distribution; floating-point formulas, no certified error or mixing bound'}
