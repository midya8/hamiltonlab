"""Fixed-setting, two-dimensional MCMC with explicit transition and work audits."""
import math
import random


def real(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(name+' must be finite')
    return float(x)


def integer(x,lo,hi,name):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(f'{name} must be integer in[{lo},{hi}]')
    return x


def vector(x,name):
    if not isinstance(x,list) or len(x)!=2:raise ValueError(name+' must have two coordinates')
    return [real(z,name) for z in x]


def evaluate(target,q):
    u,g=target(q[:]);u=real(u,'potential');g=vector(g,'gradient')
    return u,g


def kinetic(p,mass):return math.fsum(p[i]**2/(2*mass[i]) for i in range(2))


def leapfrog(target,position,momentum,step_size=.1,steps=10,mass=None):
    q=vector(position,'position');p=vector(momentum,'momentum');m=vector([1.,1.] if mass is None else mass,'mass')
    if any(z<=0 or z>1e4 for z in m):raise ValueError('mass must be in(0,1e4]')
    eps=real(step_size,'step_size');n=integer(steps,1,1000,'steps')
    if not 0<abs(eps)<=10:raise ValueError('absolute step_size must be in(0,10]')
    u,g=evaluate(target,q);h0=u+kinetic(p,m);p=[p[i]-.5*eps*g[i] for i in range(2)]
    for k in range(n):
        q=[q[i]+eps*p[i]/m[i] for i in range(2)];u,g=evaluate(target,q)
        factor=.5 if k==n-1 else 1.;p=[p[i]-factor*eps*g[i] for i in range(2)]
    h1=u+kinetic(p,m)
    if not all(math.isfinite(z) for z in q+p+[h0,h1]):raise ValueError('nonfinite trajectory')
    return {'position':q,'momentum':p,'initial_hamiltonian':h0,'final_hamiltonian':h1,'energy_error':h1-h0,
            'potential':u,'gradient_evaluations':n+1,'potential_evaluations':n+1}


def accept(log_uniform,energy_error):
    z=real(log_uniform,'log uniform');d=real(energy_error,'energy error')
    if z>0:raise ValueError('log uniform must be<=0')
    return z<min(0.,-d)


def sample(target,initial,method='hmc',draws=1000,warmup=200,step_size=.1,steps=10,mass=None,seed=0):
    q=vector(initial,'initial');n=integer(draws,2,10000,'draws');w=integer(warmup,0,10000,'warmup')
    integer(seed,0,2**32-1,'seed');eps=real(step_size,'step_size');l=integer(steps,1,1000,'steps')
    m=vector([1.,1.] if mass is None else mass,'mass')
    if any(z<=0 or z>1e4 for z in m):raise ValueError('mass must be in(0,1e4]')
    if method not in ['hmc','random_walk']:raise ValueError('unknown method')
    if not 0<eps<=10:raise ValueError('step_size must be in(0,10]')
    budget=(n+w)*(l+1 if method=='hmc' else 1)
    if budget>1000000:raise ValueError('target evaluation budget exceeds1000000')
    calls=[0]
    def counted(q):
        calls[0]+=1
        return target(q)
    rng=random.Random(seed);current,_=evaluate(counted,q);trace=[];kept=[]
    for index in range(w+n):
        before=q[:];invalid=False;error=None;proposal=None;proposed_u=None
        if method=='hmc':
            p=[rng.gauss(0.,math.sqrt(z)) for z in m]
            try:
                result=leapfrog(counted,q,p,eps,l,m);proposal=result['position'];proposed_u=result['potential'];error=result['energy_error']
            except (ValueError,OverflowError):
                invalid=True
        else:
            proposal=[q[i]+eps*rng.gauss(0.,1.) for i in range(2)]
            try:proposed_u,_=evaluate(counted,proposal);error=proposed_u-current
            except (ValueError,OverflowError):invalid=True
        uniform=rng.random()
        while uniform==0.:uniform=rng.random()
        logu=math.log(uniform)
        chosen=not invalid and accept(logu,error)
        if chosen:q=proposal[:];current=proposed_u
        if index>=w:kept.append(q[:])
        trace.append({'iteration':index+1,'warmup':index<w,'before':before,'proposal':proposal,'state':q[:],
                      'accepted':chosen,'invalid_trajectory':invalid,'energy_error':error,'log_uniform':logu})
    retained=trace[w:]
    return {'samples':kept,'transitions':trace,'method':method,'seed':seed,'draws':n,'warmup':w,'step_size':eps,
            'steps':l if method=='hmc' else None,'mass':m if method=='hmc' else None,
            'retained_acceptance_rate':sum(t['accepted'] for t in retained)/n,
            'invalid_trajectories':sum(t['invalid_trajectory'] for t in trace),
            'potential_gradient_callback_calls':calls[0],
            'semantics':'all retained transitions including repeated rejected states; fixed settings, discarded warmup is not adaptation; no convergence guarantee'}
