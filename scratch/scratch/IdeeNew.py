'''
Created on 21.05.2012

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
NP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype = 'f')

C = np.array([ PX(xi[0], xi[1]) for xi in NP ], dtype = 'f')


C1 = np.linalg.inv(C)

P_arr = np.array(P).reshape(1, 4)

N_mtx = sp.Matrix(np.dot(P_arr, C1))
N_mtx.simplify()


dN_mtx = N_mtx.jacobian([x_, y_]).T
dN_mtx.simplify()

N_fn = sp.lambdify([x_, y_], N_mtx)
dN_fn = sp.lambdify([x_, y_], dN_mtx)




print "N_mtx=", N_mtx
print "dN_mtx=", dN_mtx            # warum 2x4?
print N_fn(-1, -1)
print dN_fn(-1, -1)



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


def index_mapping_node(dNx_mtx):

    print 'dNx_mtx =\n;', dNx_mtx

    dN_idx_map_one = ((0, 1, 1, 0), slice(None))

    print 'dNx_mtx, idx = \n', dNx_mtx[dN_idx_map_one]

    B_mtx = np.arange(24, dtype = 'f').reshape(3, 2, 4)

    print 'B_mtx - orig\n', B_mtx

    B_idx_map_one = ((0, 1, 2, 2), (0, 1, 0, 1), slice(None))

    print 'B_mtx - selection\n', B_mtx[B_idx_map_one]


    B = zeros ((3, 2, 4), dtype = 'f')


    B[B_idx_map_one] = dNx_mtx[dN_idx_map_one]

    return B.reshape(3, 8)

B_mtx = index_mapping_node(b_mtx)

print "B_mtx =\n", B_mtx

E= 100000.00
v= 0.2

D_mtx = (E/(1-v**2))* np.array ([[1,v,0],[v,1,0],[0,0,((1-v)/2)]], dtype = 'f')

print D_mtx

def get_K_mtx (D_mtx, B_mtx):
    B_mtx_T = B_mtx.T
    return dot(dot(B_mtx_T,D_mtx),B_mtx)

K_mtx = get_K_mtx(D_mtx,B_mtx)
    
print "K-Matrix =\n", K_mtx