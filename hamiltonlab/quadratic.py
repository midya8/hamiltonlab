"""Per-coordinate finite leapfrog maps for separable quadratic targets."""
import math
from .core import real,integer,vector


def multiply(a,b):
    return [[math.fsum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def quadratic_map(precision,mass,step_size,steps,position=None):
    k=vector(precision,'precision');m=vector(mass,'mass');e=real(step_size,'step_size');n=integer(steps,1,1000,'steps')
    q=vector([0.,0.] if position is None else position,'position')
    if any(not 0<z<=1e4 for z in k+m):raise ValueError('precision/mass must be in(0,1e4]')
    if not 0<abs(e)<=10:raise ValueError('absolute step_size must be in(0,10]')
    coords=[]
    for i in range(2):
        a=1-e*e*k[i]/(2*m[i]);b=e/m[i];c=-e*k[i]*(1-e*e*k[i]/(4*m[i]))
        single=[[a,b],[c,a]];power=[[1.,0.],[0.,1.]]
        for _ in range(n):power=multiply(single,power)
        aa,bb=power[0];cc,dd=power[1]
        values=[z for row in power for z in row]
        if not all(math.isfinite(z) for z in values):raise ValueError('nonfinite quadratic map')
        noise=bb*bb*m[i]
        # Proposal under q~N(0,1/k), fresh p~N(0,m), before accept/reject.
        stationary_proposal_variance=aa*aa/k[i]+noise
        esjd=(aa-1)**2/k[i]+noise
        energy_defect=[[k[i]*aa*aa+cc*cc/m[i]-k[i],k[i]*aa*bb+cc*dd/m[i]],
                       [k[i]*aa*bb+cc*dd/m[i],k[i]*bb*bb+dd*dd/m[i]-1/m[i]]]
        if not all(math.isfinite(z) for z in [noise,stationary_proposal_variance,esjd]+[v for row in energy_defect for v in row]):raise ValueError('nonfinite map moments')
        coords.append({'one_step_map':single,'trajectory_map':power,'determinant':aa*dd-bb*cc,
                       'conditional_position_mean':aa*q[i],'conditional_position_variance':noise,
                       'stationary_input_proposal_variance':stationary_proposal_variance,
                       'stationary_input_expected_squared_jump':esjd,'hamiltonian_quadratic_defect':energy_defect,
                       'linear_stability_parameter':e*e*k[i]/m[i],
                       'strictly_inside_linear_stability_region':0<e*e*k[i]/m[i]<4})
    return {'precision':k,'mass':m,'step_size':e,'steps':n,'position':q,'coordinates':coords,
            'semantics':'finite-step unflipped leapfrog map for separable quadratic U; Gaussian momentum integrated analytically; all proposal moments are before Metropolis; floating-point, not a certified exact kernel or mixing bound'}
