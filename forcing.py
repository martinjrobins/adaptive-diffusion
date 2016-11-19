from __future__ import division
from sympy import *
from sympy.functions import exp


x,y = symbols('x y',real=True)
a,b = symbols('a b',real=True)

f = exp(-a*(x-0.5)**2-b*(y-0.5)**2)
pprint(f)
lf = diff(diff(f,x),x) + diff(diff(f,y),y)
pprint(simplify(lf))

#adaptx = diff(f,x)/diff(diff(f,x),x)
#adapty = diff(f,y)/diff(diff(f,y),y)
#plot(adaptx.subs(y,0.55).subs(a,50).subs(b,50),
#     adaptx.subs(y,0.55).subs(a,50).subs(b,50),
#     (x,0,1))
#
print '-----------------------------------------------------'
print '           [dfdx,dfdy]'
print '----------------------------------------------------'

gradient = Matrix([diff(f,x),diff(f,y)])
pprint(gradient)

print '-----------------------------------------------------'
print '           [df2dx2,df2dxy]'
print '           [df2dyx,df2dy2]'
print '----------------------------------------------------'
laplace = Matrix([[diff(diff(f,x),x),diff(diff(f,y),x)],
                  [diff(diff(f,x),y),diff(diff(f,y),y)]])
pprint(laplace)


print '-----------------------------------------------------'
print '           -laplace*gradient'
print '----------------------------------------------------'
adapt = -laplace*gradient
#adapt = adapt/((laplace.norm()+gradient.norm())**2)
pprint(adapt)
plot(adapt[0].subs(y,0.5).subs(a,200).subs(b,200),
    (x,0,1))
print '-----------------------------------------------------'
print '           d norm([dfdx,dfdy] dx'
print '----------------------------------------------------'
pprint(simplify(diff(gradient,x)))
print '-----------------------------------------------------'
print '           d norm([dfdx,dfdy] dy'
print '----------------------------------------------------'
pprint(simplify(diff(gradient,y)))
print '-----------------------------------------------------'
print '            fxx*fyy-fxy*fyx '
print '----------------------------------------------------'
inflection = diff(diff(f,x),x)*diff(diff(f,y),y)-diff(diff(f,x),y)*diff(diff(f,y),x)
inflection2 = diff(diff(f,x),x)*diff(diff(f,y),y)
inflection = abs(simplify(inflection))
inflection2 = abs(simplify(inflection2))
pprint(inflection)
pprint(inflection2)
plot( (f).subs(y,0.5).subs(a,4).subs(b,4),
        (gradient*inflection/(f**3*sqrt(a**2+b**2))).subs(y,0.5).subs(a,50).subs(b,50),
       (x,0,1))
print '-----------------------------------------------------'
print '            adapt force '
print '----------------------------------------------------'
pprint(simplify(gradient/f))
pprint(simplify(diff(gradient/f,x)))
print '-----------------------------------------------------'
print '            adapt force '
print '----------------------------------------------------'
pprint(simplify(diff(gradient/f,y)))

