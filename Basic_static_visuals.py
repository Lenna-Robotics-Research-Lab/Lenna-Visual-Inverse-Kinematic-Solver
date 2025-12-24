import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# DH parameters [theta, d, a, alpha] - standard 6DOF arm [file:1]
DH = np.array([
    [0, 0.15, 0.0,   np.pi/2],   # Joint 1 (base)
    [0, 0,    0.35,  0],         # Joint 2 (shoulder)
    [0, 0,    0.40,  0],         # Joint 3 (elbow)  
    [0, 0.30, 0.0,  -np.pi/2],   # Joint 4
    [0, 0,    0.25,  np.pi/2],   # Joint 5 (wrist 1)
    [0, 0.10, 0.0,  0]           # Joint 6 (wrist 2,3)
])

def rotation_matrix(theta, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha)],
        [0,              np.sin(alpha),                np.cos(alpha)]
    ])

def transformation_matrix(theta, d, a, alpha):
    T = np.eye(4)
    T[:3,:3] = rotation_matrix(theta, alpha)
    T[0,3] = a
    T[2,3] = d
    return T

def forward_kinematics(q):
    """q: 6 joint angles (rad), returns end-effector pose [x,y,z,R,P,Y]"""
    T_total = np.eye(4)
    for i in range(6):
        theta = q[i] + DH[i,0]
        d, a, alpha = DH[i,1:]
        T = transformation_matrix(theta, d, a, alpha)
        T_total = T_total @ T
    
    pos = T_total[:3,3]
    RPY = rpy_from_matrix(T_total[:3,:3])
    return np.concatenate([pos, RPY])

def rpy_from_matrix(R):
    """Extract RPY from rotation matrix"""
    pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
    if abs(pitch - np.pi/2) < 1e-6:
        roll = 0
        yaw = np.arctan2(R[1,0], R[0,0])
    elif abs(pitch + np.pi/2) < 1e-6:
        roll = 0
        yaw = -np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
        yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
    return np.array([roll, pitch, yaw])

def inverse_kinematics(target_pose, q0=np.zeros(6), max_iter=100):
    """target_pose: [x,y,z,R,P,Y], returns joint angles"""
    def pose_error(q):
        current_pose = forward_kinematics(q)
        return np.sum((current_pose - target_pose)**2)
    
    result = minimize(pose_error, q0, method='L-BFGS-B', 
                     bounds=[(-np.pi, np.pi)]*6, options={'maxiter': max_iter})
    return result.x if result.success else None

def generate_trajectory(q_start, q_end, steps=50):
    """Linear joint-space trajectory - FIXED"""
    t = np.linspace(0, 1, steps)[:, np.newaxis]  # Shape: (steps, 1)
    return q_start + t * (q_end - q_start)  # Broadcasting works correctly

def visualize_trajectory(q_traj):
    """3D visualization of robot motion"""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    link_lengths = [0, 0.35, 0.40, 0.30, 0.25, 0.10]
    
    for i, q in enumerate(q_traj[::5]):
        T = np.eye(4)
        points = [np.array([0,0,0])]
        
        for j in range(6):
            theta = q[j] + DH[j,0]
            d, a, alpha = DH[j,1:]
            T = T @ transformation_matrix(theta, d, a, alpha)
            points.append(T[:3,3].copy())
        
        points = np.array(points)
        ax.plot(points[:,0], points[:,1], points[:,2], 'b-', linewidth=2, alpha=0.7)
        if i == 0: 
            ax.scatter(points[0,0], points[0,1], points[0,2], c='g', s=100, label='Start')
        ax.scatter(points[-1,0], points[-1,1], points[-1,2], c='r', s=100, label='End' if i==len(q_traj[::5])-1 else "")
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.8,0.8]); ax.set_ylim([-0.8,0.8]); ax.set_zlim([0,1])
    plt.title('6-DOF Robot Kinematic Simulation')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main simulation - matches project requirements [file:1]
if __name__ == "__main__":
    q_current = np.zeros(6)
    print("6-DOF Robot Simulator (FK/IK + Path Planning + Visualization)")
    print("Commands: 'rel dx dy dz' or 'abs x y z R P Y' or 'quit'")
    print("Example: rel 0.1 0 0    or    abs 0.5 0.2 0.4 0 0.1 0\n")
    
    while True:
        try:
            cmd = input("Enter command: ").strip().split()
            if cmd[0].lower() == 'quit': break
            
            if cmd[0].lower() == 'rel':
                dx, dy, dz = map(float, cmd[1:4])
                current_pos = forward_kinematics(q_current)[:3]
                target_pose = np.append(current_pos + [dx,dy,dz], [0,0,0])
                print(f"Relative move: +[{dx:.2f}, {dy:.2f}, {dz:.2f}]m")
                
            elif cmd[0].lower() == 'abs':
                target_pose = np.array(list(map(float, cmd[1:])))
                print(f"Absolute target: [{target_pose[0]:.2f}, {target_pose[1]:.2f}, {target_pose[2]:.2f}, "
                      f"{target_pose[3]:.2f}, {target_pose[4]:.2f}, {target_pose[5]:.2f}]")
            
            else:
                print("Invalid command!")
                continue
            
            q_target = inverse_kinematics(target_pose, q_current)
            if q_target is not None:
                trajectory = generate_trajectory(q_current, q_target, steps=50)
                visualize_trajectory(trajectory)
                q_current = q_target
                final_pose = forward_kinematics(q_current)
                print(f"✓ Success! Final pose: [{final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}, "
                      f"{final_pose[3]:.2f}, {final_pose[4]:.2f}, {final_pose[5]:.2f}]\n")
            else:
                print("✗ IK failed - target unreachable or outside workspace\n")
                
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
