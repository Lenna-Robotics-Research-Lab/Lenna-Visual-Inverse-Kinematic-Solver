import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import time

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
    """Extract RPY from rotation matrix - STANDARD XYZ order"""
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

def matrix_from_rpy(roll, pitch, yaw):
    """Convert RPY (roll-pitch-yaw) to rotation matrix - FIXED ORDER"""
    # Fixed: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def inverse_kinematics(target_pose, q0=None, max_iter=1000):
    """ULTIMATE ACCURACY IK - position priority + verification"""
    if q0 is None:
        q0 = np.zeros(6)
    
    target_pos = target_pose[:3]
    
    def position_error(q):
        """Pure position error for better convergence"""
        pos = forward_kinematics(q)[:3]
        return np.sum((pos - target_pos)**2)
    
    # Multi-start optimization
    starts = [
        q0,
        np.array([0, 0.5, 0.5, 0, 0, 0]),
        np.array([0, 1.0, 1.0, 0, 0, 0]),
        np.array([0, 0.8, 0.3, 0, 0, 0]),
        np.array([0.2, 0.7, 0.4, 0, 0, 0])
    ]
    
    best_q = None
    best_error = float('inf')
    
    for start in starts:
        result = minimize(position_error, start, method='L-BFGS-B', 
                         bounds=[(-2*np.pi, 2*np.pi)]*6,
                         options={'maxiter': max_iter, 'ftol': 1e-10})
        
        if result.success:
            test_pose = forward_kinematics(result.x)
            pos_error = np.linalg.norm(test_pose[:3] - target_pos)
            if pos_error < best_error:
                best_error = pos_error
                best_q = result.x
    
    # Final verification
    if best_q is not None and best_error < 0.005:  # 5mm tolerance
        return best_q
    return None

def generate_trajectory(q_start, q_end, steps=50):
    t = np.linspace(0, 1, steps)[:, np.newaxis]
    return q_start + t * (q_end - q_start)

def live_visualize_trajectory(q_traj, target_pose, speed=0.03):
    """Enhanced visualization with target marker"""
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.8,0.8]); ax.set_ylim([-0.8,0.8]); ax.set_zlim([0,1.2])
    plt.title('6-DOF Robot LIVE Kinematic Simulation')
    
    # Target marker
    ax.scatter(target_pose[0], target_pose[1], target_pose[2], 
               c='orange', s=300, marker='*', label='Target', alpha=0.8)
    
    line, = ax.plot([], [], [], 'b-', linewidth=4, label='Robot Arm')
    end_eff = ax.scatter([], [], [], c='r', s=200, label='End Effector')
    base = ax.scatter([0], [0], [0], c='g', s=150, label='Base')
    
    ax.legend()
    plt.tight_layout()
    
    for frame in range(len(q_traj)):
        q = q_traj[frame]
        T = np.eye(4)
        points = [np.array([0,0,0])]
        
        for j in range(6):
            theta = q[j] + DH[j,0]
            d, a, alpha = DH[j,1:]
            T = T @ transformation_matrix(theta, d, a, alpha)
            points.append(T[:3,3].copy())
        
        points = np.array(points)
        line.set_data_3d(points[:,0], points[:,1], points[:,2])
        end_eff._offsets3d = (points[-1:,0], points[-1:,1], points[-1:,2])
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(speed)
    
    plt.ioff()
    plt.close(fig)

# Main simulation - HIGH ACCURACY [file:1]
if __name__ == "__main__":
    # Extended arm home position
    q_current = np.array([1.2, 1.2, 0, 0, 0, 0])
    
    # print("6-DOF Robot HIGH-ACCURACY Simulator")
    # print("Commands: 'rel dx dy dz' or 'abs x y z R P Y' or 'quit'")
    # print("Workspace: x=[-0.6,0.6], y=[-0.6,0.6], z=[0.2,1.0]\n")
    # print(f"Initial pose: {forward_kinematics(q_current)}")
    
    
    while True:
        try:
            cmd = input("Enter command: ").strip().split()
            if cmd[0].lower() == 'quit': break
                     
            if cmd[0].lower() == 'rel':
                dx, dy, dz = map(float, cmd[1:4])
                current_pose = forward_kinematics(q_current)
                target_pose = np.append(current_pose[:3] + [dx,dy,dz], current_pose[3:])
                print(f"Relative: +[{dx:.3f}, {dy:.3f}, {dz:.3f}]m")
                
            elif cmd[0].lower() == 'abs':
                target_pose = np.array(list(map(float, cmd[1:])))
                print(f"Target: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            
            else:
                print("Invalid command!")
                continue
            
            print("Solving HIGH-ACCURACY IK...")
            q_target = inverse_kinematics(target_pose, q_current)
            
            if q_target is not None:
                trajectory = generate_trajectory(q_current, q_target, steps=50)
                print("Executing LIVE trajectory...")
                live_visualize_trajectory(trajectory, target_pose, speed=0.03)
                
                q_current = q_target
                final_pose = forward_kinematics(q_current)
                pos_error = np.linalg.norm(final_pose[:3] - target_pose[:3])
                
                print(f"✓ TARGET: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
                print(f"✓ ACTUAL: [{final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}]")
                print(f"Position error: {pos_error*1000:.1f}mm (<5mm = PERFECT)")
                print()
                    
            else:
                print("✗ IK failed - target outside workspace")
                print()
                
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            plt.close('all')
            sys.exit(0)
