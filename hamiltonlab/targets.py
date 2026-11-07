"""Analytic synthetic targets, no fitted data or downloaded weights."""
import math
from .core import vector


def correlated(q):
    x,y=vector(q,'position');rho=.95;den=1-rho*rho
    return .5*(x*x-2*rho*x*y+y*y)/den,[(x-rho*y)/den,(y-rho*x)/den]


def mixture(q):
    x,y=vector(q,'position');left=-.5*(x+6)**2;right=-.5*(x-6)**2
    high=max(left,right);a=math.exp(left-high);b=math.exp(right-high)
    return .5*y*y-high-math.log((a+b)/2),[(a*(x+6)+b*(x-6))/(a+b),y]


def standard(q):
    x,y=vector(q,'position')
    return .5*(x*x+y*y),[x,y]


TARGETS={'standard':standard,'correlated':correlated,'mixture':mixture}
TRUTH={'correlated':{'mean':[0.,0.],'covariance':[[1.,.95],[.95,1.]]},
       'mixture':{'mean':[0.,0.],'covariance':[[37.,0.],[0.,1.]],'positive_x_probability':.5}}
