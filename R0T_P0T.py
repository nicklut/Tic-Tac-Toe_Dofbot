import numpy as np

# Input: A list of the rotations for the 5 axes, Returns: R0T, P0T
def get_R_P(q):
    # Get all the Pi-1,i values
    P01, P12, P23, P34, P45, P5T = get_P() 
    
    # Get all the Ri-1,i values
    R01, R12, R23, R34, R45 = Rz(q[0]), Ry(-q[1]), Ry(-q[2]), Ry(-q[3]), Rx(-q[4])
    # Find R0T = R01*R12*R34*R45*R5T -> R5T is the identity matrix so it is excluded
    R0T = np.matmul(np.matmul(np.matmul(np.matmul(R01,R12),R23),R34),R45) 

    # Calculate P0T 
    R02 = np.matmul(R01, R12)
    R03 = np.matmul(R02, R23)
    R04 = np.matmul(R02, R34)
    P0T = P01 + np.matmul(R01, P12) + np.matmul(R02, P23) + np.matmul(R03, P34) \
        + np.matmul(R04, P45) + np.matmul(R0T, P5T)
    
    return np.around(R0T, decimals=4), np.round(P0T, decimals=4)

# Input: theta in degrees, Output: The respective rotation matrix for the given direciton
def Rx(theta_degrees): 
    theta = theta_degrees*(np.pi/180)
    return np.array([[1,0,0],[0, np.cos(theta), -1*np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def Ry(theta_degrees): 
    theta = theta_degrees*(np.pi/180)
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)] ])

def Rz(theta_degrees): 
    theta = theta_degrees*(np.pi/180)
    return np.array([[np.cos(theta), -1*np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])

# This function uses the given l values and direction to find the P values. 
def get_P():
    l = np.array([61, 43.5, 82.85, 82.85, 73.85, 54.57])*0.001
    ex = np.array([1,0,0])
    ez = np.array([0,0,1])
    
    return l[0]*ez, l[1]*ez, l[2]*ex, -l[3]*ez, -(l[4]+l[5])*ex
    
if __name__ == "__main__": 
    #R0T, P0T = get_R_P([90,90,90,90,90])
    #R0T, P0T = get_R_P([0,0,0,0,0])
    R0T, P0T = get_R_P([0,45, 135,45,135])
    
    print("R0T:")
    print(R0T)
    print("P0T:")
    print(P0T)