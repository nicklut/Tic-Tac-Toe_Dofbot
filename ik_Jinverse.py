import numpy as np
import general_robotics_toolbox as rox

def ik_Jinverse(robot,q0,Rd,Pd,Nmax,alpha,tol,J):
    
    n = (q0).shape[0]
    T = rox.fwdkin(robot,q0)
    R = T.R
    P = T.p
    
    q = np.zeros((n, Nmax+1))
    q[:,0] = q0 # output joint displacements
    p0T = np.zeros((3,Nmax+1)) # output p
    RPY0T = np.zeros((3,Nmax+1)) # output Euler angles
    
    iternum = 0
    
    # get the pose error
    dR = np.matmul(R, Rd.T)
    dX = np.concatenate((np.array(rox.R2rpy(dR))[None].T, np.reshape(P-Pd, (3,1))))
    
    # iterative update    
    while (dX>tol).any():
        if iternum <= Nmax:
            # forward kinematics
            p0T[:,iternum] = rox.fwdkin(robot, q[:, iternum]).p
            R = rox.fwdkin(robot, q[:, iternum]).R
            
            Jq = rox.robotjacobian(robot, q[:, iternum])
            RPY0T[:, iternum] = np.reshape(np.array(rox.R2rpy(R))[None].T, (3,))
            
            # get the pose error
            dR = np.matmul(R,Rd.T)
            dX = np.concatenate((np.array(rox.R2rpy(dR))[None].T, np.reshape(p0T[:, iternum]-Pd, (3,1))))
            
            # Jacobian update       
            # dX = [beta s; P-Pd]
            np.copyto(q[:,iternum+1],q[:,iternum]-alpha*np.reshape(np.matmul(np.linalg.pinv(Jq), dX), (n,)))
            iternum = iternum + 1
            
        else:
            break
    
    iternum = iternum - 1
    
    q, p0T, RPY0T = q[:, :iternum], p0T[:, :iternum], RPY0T[:, :iternum]
    
    return q
    
def Define_Dofbot(): 
    l0 = 0.061
    l1 = 0.0435
    l2 = 0.08285
    l3 = 0.08285
    l4 = 0.07385
    l5 = 0.05457
    
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    DofbotP = np.array([(l0+l1)*ez, np.zeros((3,)), l2*ex, -l3*ez, np.zeros((3,)), -(l4+l5)*ex]).T
    DofbotH = np.array([ez, -ey, -ey, -ey, -ex]).T
    Dofbot_joint_type = np.array([0,0,0,0,0])

    return rox.Robot(DofbotH, DofbotP, Dofbot_joint_type)

if __name__ == "__main__":
    R0T = np.array([[-0.75, -0.1047, -0.6531], [-0.433, 0.8241, 0.3652], [0.5, 0.5567, -0.6634]])
    P0T = np.array([0.2058, 0.1188, 0.1464])
    q0 = np.array([25, 50, 75, 30, 30])*(np.pi/180)

    tol = np.array([0.02, 0.02, 0.02, 0.001, 0.001, 0.001])
    Nmax = 200
    epsilon = 0.1 
    alpha = 0.1

    initialize_Dofbot = Define_Dofbot()
    J = rox.robotjacobian(initialize_Dofbot, q0)
    q = ik_Jinverse(initialize_Dofbot,q0,R0T,P0T,Nmax,alpha,tol,J)

    print(np.rad2deg(q[:, -1]))