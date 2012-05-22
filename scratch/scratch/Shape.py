'''
Created on 10.05.2012

@author: demian
'''
import numpy as np
import sympy as sp

from scipy.linalg import \
     inv, det
from numpy import \
     array, zeros, int_, float_, ix_, dot, linspace, hstack, vstack, arange, \
     identity, linalg 
     
     
x_, y_, z_, t_, s_, r_ = sp.symbols('x,y,z,t,s,r')



def Shape(Xex,Xey):
    
    Xe = np.array ([Xex,Xey])
    s_,r_ = sp.symbols('s,r')
    
    r_N = np.array ([0.25,0.25,0.25,0.25], dtype = 'f')
   

    h_N = np.array ([.25,-0.25,0.25,-0.25], dtype = 'f')
   
    
    
    gs_N = np.array([-0.25,0.25,0.25,-0.25], dtype = 'f')
   
    
    
    gr_N = np.array([-0.25,-0.25,0.25,0.25])
    

    
    N_t_Xe = (np.dot(r_N,Xe) + np.dot((s_*gs_N),Xe) + np.dot((r_*gr_N),Xe)+np.dot((s_*r_*h_N),Xe))
    
    
    print "Xex=", Xex
    print "Xey=", Xey
    print "N_t_Xe=", N_t_Xe
    
    return N_t_Xe





''' 1. Beispiel'''
print "erstes Beispiel"

Xex = np.array([0.,3.,2.,-1.]).reshape (4,1)
Xey = np.array([0.,1.,4.,2.]).reshape (4,1)

N_t_Xe = Shape(Xex, Xey)

N_t_Xe_r=  sp.Matrix(N_t_Xe)

print "N_t_Xe_r=", N_t_Xe_r

N_t_Xe_t = N_t_Xe_r.T

print "N-t_Xe_t=",N_t_Xe_t

Jac_N=N_t_Xe_t.jacobian([s_,r_]) #lokale Ableitungen , jetzt invJ lokale Ableitungen

print "Jac_N" ,Jac_N

Jac_X= sp.lambdify([r_,s_],Jac_N)

Jac_O= sp.Matrix(Jac_X (0,0))


print "Jac_O=", Jac_O



g = np.array([[-0.25, -0.25],[0.25, -0.25],[0.25,0.25],[-0.25,0.25]])

Jac_inv = linalg.inv(Jac_O)
 

print "Jac_inv =", Jac_inv


print "g=",g




b = np.dot(g,Jac_inv)
print "b=", b
def index_mapping_four_Node (dNx_mtx):
    
    B_idx_map_four = ((0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2),(0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7))
     
    dN_idx_map_four =((0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3),(0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0))
     
    B = zeros ((3,8),dtype ='f')
          
    B = zeros ((3,8),dtype ='f')

    B[B_idx_map_four] = dNx_mtx[dN_idx_map_four]
    
    return B

B_lin = index_mapping_four_Node (b)

print "B-Matrix = ", B_lin

     


''' Ende Beispiel'''

'''
def Jacobian(x):
    
    N_t_Xe = sp.Matrix(x)
    N_t_Xe.simplify()
    
    dN_mtx = N_t_Xe.jacobian([s_, r_]).T
    
    print "dN_mtx", dN_mtx
    
    return dN_mtx
    
'''



'''
P = [1, x_, y_, x_ * y_]
PX = sp.lambdify([x_, y_], P)

# Nodal points
NP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype = 'f')

C = np.array([ PX(xi[0], xi[1]) for xi in NP ], dtype = 'f')


C1 = np.linalg.inv(C)

P_arr = np.array(P).reshape(1, 4)
print P
print P_arr


N_mtx = sp.Matrix(np.dot(P_arr, C1)) # dot multiplikation
N_mtx.simplify()

dN_mtx = N_mtx.jacobian([x_, y_]).T
dN_mtx.simplify()

N_fn = sp.lambdify([x_, y_], N_mtx)
dN_fn = sp.lambdify([x_, y_], dN_mtx)

print N_mtx
print dN_mtx
print N_fn(-1, -1)
print dN_fn(-1, -1)


P = [1, x_, y_, x_ * y_]
PX = sp.lambdify([x_, y_], P)

# Nodal points
NP = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype = 'f')
CD = 2*NP
print CD
C = np.array([ PX(xi[0], xi[1]) for xi in NP ], dtype = 'f')
print C
'''