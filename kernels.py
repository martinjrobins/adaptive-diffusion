from __future__ import division
from sympy import *
from sympy.plotting import plot

from sympy.functions import exp
from sympy.core.containers import Tuple


xi = symbols(r'x_i',real=True)
yi = symbols(r'y_i',real=True)

xj = symbols(r'x_j',real=True)
yj = symbols(r'y_j',real=True)

hi = symbols(r'h_i',positive=true)
hj = symbols(r'h_j',positive=true)
h = 0.5*(hi+hj)

ri = Matrix([xi,yi])
rj = Matrix([xj,yj])
dx = ri-rj


d = symbols(r'd',positive=true)
k = symbols(r'k',positive=true)

#kij = exp(-dx.dot(dx)/(h**2))
kij = Piecewise(((2-dx.norm()/h)**4*(1 + 2*dx.norm()/h),dx.norm()<2*h),(0,True))
print '---------------------------------------'
print '           k_ij'
print '---------------------------------------'
pprint(simplify(kij))

print '-----------------------------------------------------'
print '           du/dt = d^2k_ijdx^2 + d^2k_ijdy^2'
print '----------------------------------------------------'
laplace = diff(diff(kij,xi),xi) + diff(diff(kij,yi),yi)
pprint(simplify(laplace))

print '---------------------------------------'
print '           dx/dt = force '
print '---------------------------------------'
delta = d-dx.norm()
pot = k*delta**2
f = Matrix([diff(pot,xi),diff(pot,yi)])
delta = d-dx.norm()
pot = k*delta**2
f = Matrix([diff(pot,xi),diff(pot,yi)])
pprint(simplify(f))
