'''
Created on 22.05.2012

@author: demian
'''

import numpy as np
import sympy as sp
from scipy.linalg import \
     inv, det
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity
     
x_, y_, z_, t_ = sp.symbols('x,y,z,t')

P = [1, x_, y_, x_ * y_]
PX = sp.lambdify([x_, y_], P)

# Nodal points
NP = np.array([[0, 0], [3, 1], [2, 4], [-1, 2]], dtype = 'f')

C = np.array([ PX(xi[0], xi[1]) for xi in NP ], dtype = 'f')


C1 = np.linalg.inv(C)

P_arr = np.array(P).reshape(1, 4)

N_mtx = sp.Matrix(np.dot(P_arr, C1))
N_mtx.simplify()


dN_mtx = N_mtx.jacobian([x_, y_]).T
dN_mtx.simplify()

N_fn = sp.lambdify([x_, y_], N_mtx)
dN_fn = sp.lambdify([x_, y_], dN_mtx)




#print "N_mtx=", N_mtx
#print "dN_mtx=", dN_mtx            # warum 2x4?
print N_fn(0, 0)
print dN_fn(-0, -0)



gaus= np.array([[0,0],[-0.77459,-0.77459],[0.77459,-0.77459],[0.77459,0.77459],[-0.77459,0.77459]], dtype ='f')


X_mtx = np.array([[0,0],[3,1],[2,4],[-1,2]], dtype ='f') #Beispiel
b_mtx = zeros ((2,4), dtype = 'f')

for i in range (0,4):
    dN = dN_fn(gaus[i,0],gaus[i,1])
    J_mtx = dot(dN,X_mtx)
    J_mtx_inv = inv(J_mtx)
    b_mtx = b_mtx + dot(J_mtx_inv,dN)
print "b_mtx = \n",b_mtx

print b_mtx[1,1]