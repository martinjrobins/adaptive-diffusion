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
r = symbols(r'r',positive=true)
h = 0.5*(hi+hj)

ri = Matrix([xi,yi])
rj = Matrix([xj,yj])
dx = ri-rj


d = symbols(r'd',positive=true)
k = symbols(r'k',positive=true)

#kij = exp(-dx.dot(dx)/(h**2))
kij = Piecewise(((2-dx.norm()/h)**4*(1 + 2*dx.norm()/h),dx.norm()<2*h),(0,True))
kijr = Piecewise(((2-r/h)**4*(1 + 2*r/h),r<2*h),(0,True))
print '---------------------------------------'
print '           k_ij'
print '---------------------------------------'
pprint(kij)
print latex(kij)
pprint(simplify(kij))

#pprint(kijr)
#xeqi = kij.subs(hi,0.075).subs(hj,0.075).subs(xj,0.5).subs(yj,0.5).subs(yi,0.5)
#plot(xeqi,(xi,-1,2),adaptive=False,nb_of_points=500)

laplace = diff(diff(kij,xi),xi) + diff(diff(kij,yi),yi)
#pprint(simplify(laplace))
print 'check if Im right'
norm = sqrt((xi-xj)**2+(yi-yj)**2)
norm2 = (xi-xj)**2+(yi-yj)**2
simple = 65536.0*((hi+hj-norm)**2)*(0.001953125*(-2.5*hi-2.5*hj+10*norm) + 0.0048828125*(hi+hj-norm) - 0.009765625*(hi+hj-norm))/(((hi+hj)**5))
#pprint(simplify(expand(laplace-simple)))

xeqi = laplace.subs(hi,0.075).subs(hj,0.075).subs(xi,0.5).subs(yi,0.5).subs(yj,0.500000001)
xeqi2 = simple.subs(hi,0.075).subs(hj,0.075).subs(xi,0.5).subs(yi,0.5).subs(yj,0.500000001)
plot(xeqi,xeqi2,(xj,0.3,0.7),adaptive=False,nb_of_points=1000)


print '-----------------------------------------------------'
print '           [dfdx,dfdy]'
print '----------------------------------------------------'

gradient = Matrix([diff(kij,xi),diff(kij,yi)])
pprint(simplify(gradient))

print '-----------------------------------------------------'
print '           [df2dx2,df2dxy]'
print '           [df2dyx,df2dy2]'
print '----------------------------------------------------'
laplace = Matrix([[diff(diff(kij,xi),xi),diff(diff(kij,yi),xi)],
                  [diff(diff(kij,xi),yi),diff(diff(kij,yi),yi)]])
pprint(simplify(laplace))


print '-----------------------------------------------------'
print '           -laplace*gradient'
print '----------------------------------------------------'
adapt = -laplace*gradient
#adapt = adapt/((laplace.norm()+gradient.norm())**2)
pprint(simplify(adapt))


print '-----------------------------------------------------'
print '           norm([dk_ijdx,dk_ijdy]'
print '----------------------------------------------------'

gradient = Matrix([diff(kij,xi),diff(kij,yi)]).norm()
gradient = simplify(gradient)
pprint(gradient)

print '-----------------------------------------------------'
print '           d norm([dk_ijdx,dk_ijdy] dx'
print '----------------------------------------------------'

pprint(simplify(diff(gradient,xi)))

print '-----------------------------------------------------'
print '           d norm([dk_ijdx,dk_ijdy] dy'
print '----------------------------------------------------'

pprint(simplify(diff(gradient,yi)))



print '-----------------------------------------------------'
print '           dk_ijdx'
print '----------------------------------------------------'

pprint(simplify(diff(kij,xi)))
pprint(simplify(diff(kij,xj)))

print '-----------------------------------------------------'
print '           dk_ijdy'
print '----------------------------------------------------'

pprint(simplify(diff(kij,yi)))
pprint(simplify(diff(kij,yj)))


print '-----------------------------------------------------'
print '           du/dt = d^2k_ijdx^2 + d^2k_ijdy^2'
print '----------------------------------------------------'

laplace = diff(diff(kij,xi),xi) + diff(diff(kij,yi),yi)
print 'check if Im right'
simple = 65536*(hi+hj-sqrt((xi-xj)**2+(yi-yj)**2))*((xi-xj)**2+(yi-yj)**2)*(0.001953125*(-2.5*hi-2.5*hj+10*sqrt((xi-xj)**2+(yi-yj)**2)) - 0.0048828125*(hi+hj-sqrt((xi-xj)**2+(yi-yj)**2)))/((hi+hi+hj)**5*((xi-xj)**2+(yi-yj)**2))
#pprint(simplify(laplace-simple))

laplacer = diff(diff(kijr,r),r)
pprint(simplify(laplacer))
print latex(simplify(laplace))
pprint(2*simplify(laplacer.subs(r,0)))
pprint(2*simplify(laplacer).subs(r,0).subs(hi,0.075).subs(hj,0.075))

pprint(simplify(laplace.subs(xj,0.5000000000123).subs(yj,0.5).subs(yi,0.5).subs(xi,0.5)))
pprint(simplify(laplace.subs(xj,0.5000000000123).subs(yj,0.50000000123).subs(yi,0.5).subs(xi,0.5).subs(hi,0.075).subs(hj,0.075)))

pprint(simplify(laplace))
xeqi = laplace.subs(hi,0.075).subs(hj,0.075).subs(xi,0.5).subs(yi,0.5).subs(yj,0.5)
xeqi2 = simple.subs(hi,0.075).subs(hj,0.075).subs(xi,0.5).subs(yi,0.5).subs(yj,0.5)
plot(xeqi,xeqi2,(xj,0.4,0.6),adaptive=False,nb_of_points=1000)

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
