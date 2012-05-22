'''
Created on 18.05.2012

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

J_mtx = dot(dn_fn(0,0),Xex)




print "Test",dN_fn(test[0,0],test[1,1])

'''
Map the local coords to global
@param r_pnt: local coords
@param X_mtx: matrix of the global coords of geo nodes
'''

def get_J_mtx(dN_fn, X_mtx):
    return dot(dN_fn(0,0), X_mtx)

def index_mapping_four_Node (dNx_mtx):
    
    B_idx_map_four = ((0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2),(0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7))
     
    dN_idx_map_four =((0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),(0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0))
     
    B = zeros ((3,8),dtype ='f')
    
    B[B_idx_map_four] = dNx_mtx[dN_idx_map_four]
    
    return B

def get_B_mtx (dN_fn, X_mtx):
    J_mtx = get_J_mtx (dN_fn, X_mtx)
    J_mtx_inv= inv(J_mtx)
    dNr_mtx = np.array ([[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]], dtype ='f')
    dNx_mtx = dot(dNr_mtx,J_mtx_inv)
    print "dNx_mtx = " ,dNx_mtx
    '''
    Bx_mtx = zeros ((3,8),dtype ='f')
        
    for i in range(0,4):
        Bx_mtx [0,i] = dNx_mtx[i,0]
        Bx_mtx [1,i+4] = dNx_mtx [i,1]
        Bx_mtx [2,i] = dNx_mtx [i,1]
        Bx_mtx [2,i+4] = dNx_mtx [i,0]
    '''
    Bx_mtx = index_mapping_four_Node (dNx_mtx)
    return Bx_mtx

def run_exemple():
    print "---------Beispiel Anfang------------"
    
    X_mtx = np.array([[0,0],[3,1],[2,4],[-1,2]], dtype ='f')
    print "X_mtx", X_mtx
    
    print "Jakobi-Matrix = ", get_J_mtx (dN_fn, X_mtx)
    
    
    print "B-Matrix = ", get_B_mtx(dN_fn, X_mtx)
    
    print "---------Beispiel Ende-------------"

print run_exemple()
