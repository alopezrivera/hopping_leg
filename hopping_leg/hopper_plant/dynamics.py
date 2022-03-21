from sympy import *
from sympy.physics.vector import dynamicsymbols
init_printing()



# No Contact:

# g, L1,L2, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1, Ixx2, Ixy2, Ixz2, Iyy2, Iyz2, Izz2, m0, m1, m2, cx1,cy2,cz1, cx2, cy2, cz2 = symbols("g L_1 L_2 I_xx1 I_xy1 I_xz1 I_yy1 I_yz1 I_zz1 I_xx2 I_xy2 I_xz2 I_yy2 I_yz2 I_zz2 m_0 m_1 m_2 c_1x c_1y c_1z c_2x c_2y c_2z", constant=True)
g, L1,L2, Ixx1, Ixy1, Ixz1, Iyy1, Iyz1, Izz1, Ixx2, Ixy2, Ixz2, Iyy2, Iyz2, Izz2, m0, m1, m2, cx1,cy2,cz1, cx2, cy2, cz2 = symbols("g L1 L2 Ixx1 Ixy1 Ixz1 Iyy1 Iyz1 Izz1 Ixx2 Ixy2 Ixz2 Iyy2 Iyz2 Izz2 m0 m1 m2 cx1 cy2 cz1 cx2 cy2 cz2", constant=True)

th1, th2, q0 = dynamicsymbols("theta1 theta2 q0")
Tau0, Tau1, Tau2 = symbols("tau0 tau1 tau2")

dth1 = th1.diff("t")
dth2 = th2.diff("t")
ddth1 = dth1.diff("t")
ddth2 = dth2.diff("t")
dq0 = q0.diff("t")
ddq0 = dq0.diff("t")
# dx = x.diff("t")
# dy = y.diff("t")
# ddx = dx.diff("t")
# ddy = dy.diff("t")

def rotationMat(axis, angle):
    R = Matrix([[axis[0]**2*(1-cos(angle))+cos(angle),              axis[0]*axis[1]*(1-cos(angle))-axis[2]*sin(angle), axis[0]*axis[2]*(1-cos(angle))+axis[1]*sin(angle)],
                [axis[1]*axis[0]*(1-cos(angle))+axis[2]*sin(angle), axis[1]**2*(1-cos(angle))+cos(angle),              axis[1]*axis[2]*(1-cos(angle))-axis[0]*sin(angle)],
                [axis[2]*axis[0]*(1-cos(angle))-axis[1]*sin(angle), axis[2]*axis[1]*(1-cos(angle))+axis[0]*sin(angle), axis[2]**2*(1-cos(angle))+cos(angle)]])
    return R


def transMat(rotMat, transVec, short = False):
    T = rotMat.col_insert(3,transVec)
    if not short:
        T = T.row_insert(3,Matrix([[0,0,0,1]]))
    return T

def reshapeTransMat(transMat):
    T = transMat.col_del(3)
    return T    


# T = sum(Ti)
# Ti = 1/2 mi vi0.T vi0 + 1/2 wi0.T Ii0 wi0
# mit vi0 = J[:i] * dq[:i].T
    # mit J[:i] = (J1,j=1|J2,j=2, ... Ji,j=j)
        # mit Jij = (RotMat * (001)) x (ci - tj)
            # mit ci = E3|0 * TransMat * ti
            # mit tj = translation to global origin
    
    
    
def massInertia(Ixx,Ixy,Ixz,Iyy,Iyz,Izz):
    I = Matrix([[Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]])
    return I

# gravity vector
g0 = Matrix([g,0,0])
# joint rotation axis
jra = Matrix([0,0,1])
# prismatic joint direction
prismatic = Matrix([-1,0,0])

t0 = Matrix([0,0,0])
J0 = rotationMat([1,0,0],0) * Matrix([0,0,1])
v0 = J0 * dq0
Jw0 = Matrix([0,0,0])
c0 = transMat(rotationMat(prismatic,0),t0,True)*((q0*prismatic).row_insert(3,Matrix([1])))

T0 = m0/2 *v0.T *v0

t1 = q0 * prismatic + t0
# t1 = Matrix([0,0,0]) # t
c1 = transMat(rotationMat(jra,th1), t1,True)*Matrix([cx1,cy2,cz1,1]) #R|t * cii|1

J11 = (rotationMat(jra,th1) * Matrix([0,0,1])).cross(c1-t1)  # R * 0,0,1 x (c - t)
J1 = Matrix([[J0[0], J11[0]],
             [J0[1], J11[1]],
             [J0[2], J11[2]]])
v1 = J1 * Matrix([dq0, dth1]) 


Jw10 = rotationMat(jra,th1) * Matrix([0,0,1]) # R * 0,0,1
Jw1 = Matrix([[Jw0[0], Jw10[0]],
              [Jw0[1], Jw10[1]],
              [Jw0[2], Jw10[2]]])

w1 = Jw1 * Matrix([dq0, dth1])

I1 = massInertia(Ixx1,Ixy1,Ixz1,Iyy1,Iyz2,Izz1)



T1 = m1/2 *v1.T *v1 + w1.T/2 * I1 * w1

t2 = Matrix([L1*cos(th1),L1*sin(th1),0]) + t1 # t
c2 = transMat(rotationMat(jra,th1+th2), t2,True)*Matrix([cx2,cy2,cz2,1]) # cii|1

J12 = (rotationMat(jra,th1) * Matrix([0,0,1])).cross(c2-t1)
J22 = (rotationMat(jra,th1+th2) * Matrix([0,0,1])).cross(c2-t2) 
J2 = Matrix([[J0[0], J12[0], J22[0]],
             [J0[1], J12[1], J22[1]],
             [J0[2], J12[2], J22[2]]])

v2 = J2 * Matrix([dq0,dth1,dth2]) 

Jw20 = rotationMat(jra,th1+th2) * Matrix([0,0,1]) # R * 0,0,1
Jw2 = Matrix([[Jw0[0], Jw10[0], Jw20[0]],
              [Jw0[1], Jw10[1], Jw20[1]],
              [Jw0[2], Jw10[2], Jw20[2]]])

w2 = Jw2 * Matrix([dq0,dth1,dth2]) 
I2 = massInertia(Ixx2,Ixy2,Ixz2,Iyy2,Iyz2,Izz2)

T2 = m2/2 *v2.T *v2 + w2.T/2 * I2 * w2

V0 = -g0.T * c0 * m0
V1 = -g0.T * c1 * m1
V2 = -g0.T * c2 * m2

L = T0+T1+T2-V0-V1-V2

tau0 = simplify((L.diff(q0.diff("t"))).diff("t") - L.diff(q0))
tau1 = simplify((L.diff(th1.diff("t"))).diff("t") - L.diff(th1))
tau2 = simplify((L.diff(th2.diff("t"))).diff("t") - L.diff(th2))
tau = tau0.row_insert(1,tau1).row_insert(2,tau2)
tau



Q0,Th1,Th2,dQ0,dTh1,dTh2,ddQ0,ddTh1,ddTh2 = symbols("q0 q1 q2 dq0 dq1 dq2 ddq0 ddq1 ddq2")
tauS = tau.subs(ddq0,ddQ0).subs(ddth1,ddTh1).subs(ddth2,ddTh2).subs(dq0,dQ0).subs(dth1,dTh1).subs(dth2,dTh2).subs(q0,Q0).subs(th1,Th1).subs(th2,Th2)
## forward dynamics:
eq0 = Eq(tauS[0], Tau0)
eq1 = Eq(tauS[1], Tau1)
eq2 = Eq(tauS[2], Tau2)
# solve([eq0, eq1,eq2],[ddQ0,ddTh1,ddTh2])

pycode(solve(eq0,ddQ0))
pycode(solve(eq1,ddTh1))
pycode(solve(eq2,ddTh2))

pycode(tauS[0])
pycode(tauS[1])
pycode(tauS[2])



M = Matrix([[m0+m1+m2,0,0],
            [0,(Izz1 + Izz2 + L1**2*m2 + 2*L1*cx2*m2*cos(th2) - 2*L1*cy2*m2*sin(th2) + cx2**2*m1 + cy2**2*m1 + cx2**2*m2 + cy2**2*m2), (Izz2 + L1*cx2*m2*cos(th2) - L1*cy2*m2*sin(th2) + cx2**2*m2 + cy2**2*m2)],
            [0,(Izz2 + L1*cx2*m2*cos(th2) - L1*cy2*m2*sin(th2) + cx2**2*m2 + cy2**2*m2),(Izz2 + cx2**2*m2 + cy2**2*m2)]])

C = Matrix([[0,0,0],
            [0,dth2*(-2*L1*cx2*m2*sin(th2) - 2*L1*cy2*m2*cos(th2)),dth2*(-L1*cx2*m2*sin(th2) - L1*cy2*m2*cos(th2))],
            [0,dth1*(L1*cx2*m2*sin(th2) + L1*cy2*m2*cos(th2)),0]])

G = Matrix([g*(m0 + m1 + m2),
            L1*g*m2*sin(th1) + cx1*g*m1*sin(th1) + cy1*g*m1*cos(th1) + cx2*g*m2*sin(th1 + th2) + cy2*g*m2*cos(th1 + th2),
            cx2*g*m2*sin(th1 + th2) + cy2*g*m2*cos(th1 + th2)])

tau = (M*Matrix([ddq0,ddth1,ddth2]) + C * Matrix([dq0,dth1,dth2]) + G )



MS = M.subs(ddq0,ddQ0).subs(ddth1,ddTh1).subs(ddth2,ddTh2).subs(dq0,dQ0).subs(dth1,dTh1).subs(dth2,dTh2).subs(q0,Q0).subs(th1,Th1).subs(th2,Th2)
CS = C.subs(ddq0,ddQ0).subs(ddth1,ddTh1).subs(ddth2,ddTh2).subs(dq0,dQ0).subs(dth1,dTh1).subs(dth2,dTh2).subs(q0,Q0).subs(th1,Th1).subs(th2,Th2)
GS = G.subs(ddq0,ddQ0).subs(ddth1,ddTh1).subs(ddth2,ddTh2).subs(dq0,dQ0).subs(dth1,dTh1).subs(dth2,dTh2).subs(q0,Q0).subs(th1,Th1).subs(th2,Th2)


Minv = simplify(MS**(-1))


x = L1 * cos(th1) + L2*cos(th1+th2) - q0
y = L1 * sin(th1) + L2*sin(th1+th2)
J = Matrix([[x.diff(q0), x.diff(th1), x.diff(th2)],
            [y.diff(q0), y.diff(th1), y.diff(th2)]])

JS = J.subs(ddq0,ddQ0).subs(ddth1,ddTh1).subs(ddth2,ddTh2).subs(dq0,dQ0).subs(dth1,dTh1).subs(dth2,dTh2).subs(q0,Q0).subs(th1,Th1).subs(th2,Th2)

