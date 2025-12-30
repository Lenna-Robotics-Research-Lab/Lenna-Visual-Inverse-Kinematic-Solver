import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import time



# DH parameters [theta, d, a, alpha] - standard 6DOF arm
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



def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range for proper error computation"""
    return np.arctan2(np.sin(angle), np.cos(angle))



def orientation_error(rpy_current, rpy_target):
    """Compute orientation error with proper angle wrapping"""
    error = rpy_target - rpy_current
    # Wrap each angle difference to [-pi, pi]
    return np.array([wrap_angle(error[0]), wrap_angle(error[1]), wrap_angle(error[2])])



def inverse_kinematics(target_pose, q0=None, max_iter=2000, position_weight=10.0, orientation_weight=0.01):
    """
    Full 6-DOF IK solver - POSITION PRIORITY with relaxed orientation
    
    Parameters:
    - target_pose: [x, y, z, roll, pitch, yaw]
    - q0: initial joint configuration
    - position_weight: weight for position error (default=100.0 - HIGH PRIORITY)
    - orientation_weight: weight for orientation error (default=0.01 - VERY RELAXED)
    """
    if q0 is None:
        q0 = np.zeros(6)
    
    target_pos = target_pose[:3]
    target_rpy = target_pose[3:] if len(target_pose) == 6 else np.zeros(3)
    
    def full_pose_error(q):
        """Combined position + orientation error with STRONG position priority"""
        current_pose = forward_kinematics(q)
        
        # Position error (Euclidean distance squared)
        pos_error = np.sum((current_pose[:3] - target_pos)**2)
        
        # Orientation error (with angle wrapping) - HIGHLY RELAXED
        orient_err = orientation_error(current_pose[3:], target_rpy)
        orient_error = np.sum(orient_err**2)
        
        # Weighted sum: position HIGHLY prioritized (100:0.01 ratio)
        return position_weight * pos_error + orientation_weight * orient_error
    
    # Expanded multi-start optimization with more diverse initial guesses
    starts = [
        q0,
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 0.5, 0.5, 0, 0, 0]),
        np.array([0, 1.0, 1.0, 0, 0, 0]),
        np.array([0, 0.8, 0.3, 0, 0, 0]),
        np.array([0.2, 0.7, 0.4, 0, 0, 0]),
        np.array([np.pi/4, 0.6, 0.6, 0, 0, 0]),
        np.array([-np.pi/4, 0.6, 0.6, 0, 0, 0]),
        np.array([0, 1.2, 0.8, 0, 0, 0]),
        np.array([0.5, 0.5, 0.5, 0, 0, 0])
    ]
    
    best_q = None
    best_error = float('inf')
    
    for start in starts:
        result = minimize(full_pose_error, start, method='SLSQP', 
                         bounds=[(-2*np.pi, 2*np.pi)]*6,
                         options={'maxiter': max_iter, 'ftol': 1e-9})
        
        if result.success or result.fun < best_error:
            test_pose = forward_kinematics(result.x)
            pos_error = np.linalg.norm(test_pose[:3] - target_pos)
            
            # Evaluate based primarily on position error
            if pos_error < best_error:
                best_error = pos_error
                best_q = result.x
    
    # RELAXED verification - focus on position accuracy only
    if best_q is not None:
        final_pose = forward_kinematics(best_q)
        pos_error = np.linalg.norm(final_pose[:3] - target_pos)
        # Increased tolerance from 10mm to 20mm for better success rate
        if pos_error < 0.02:  
            return best_q
    
    # Fallback: try again with even more relaxed orientation weight
    print("  Retrying with zero orientation weight...")
    for start in starts:
        result = minimize(
            lambda q: position_weight * np.sum((forward_kinematics(q)[:3] - target_pos)**2),
            start, method='SLSQP', 
            bounds=[(-2*np.pi, 2*np.pi)]*6,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        
        if result.success or result.fun < best_error:
            test_pose = forward_kinematics(result.x)
            pos_error = np.linalg.norm(test_pose[:3] - target_pos)
            
            if pos_error < best_error:
                best_error = pos_error
                best_q = result.x
    
    if best_q is not None and best_error < 0.02:
        return best_q
    
    return None



def generate_trajectory(q_start, q_end, steps=50):
    t = np.linspace(0, 1, steps)[:, np.newaxis]
    return q_start + t * (q_end - q_start)



def live_visualize_trajectory(q_traj, target_pose, speed=0.03, hold_time=5.0):
    """Enhanced visualization with SINGLE TEXT BOX showing position and orientation"""
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.8,0.8]); ax.set_ylim([-0.8,0.8]); ax.set_zlim([0,1.2])
    plt.title('6-DOF Robot LIVE Kinematic Simulation (Full Pose Control)')
    
    # Target marker
    ax.scatter(target_pose[0], target_pose[1], target_pose[2], 
               c='orange', s=300, marker='*', label='Target', alpha=0.8)
    
    line, = ax.plot([], [], [], 'b-', linewidth=4, label='Robot Arm')
    end_eff = ax.scatter([], [], [], c='r', s=200, label='End Effector')
    base = ax.scatter([0], [0], [0], c='g', s=150, label='Base')
    
    # END EFFECTOR TRAJECTORY PATH
    traj_line, = ax.plot([], [], [], 'r--', linewidth=2, alpha=0.7, label='End Effector Path')
    
    # Single label for position and orientation
    pose_label = None
    
    ax.legend()
    plt.tight_layout()
    
    # Pre-compute end effector positions for trajectory line
    end_effector_positions = []
    
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
        
        # Store end effector position for trajectory
        end_effector_positions.append(points[-1].copy())
        
        # Update robot arm
        line.set_data_3d(points[:,0], points[:,1], points[:,2])
        end_eff._offsets3d = (points[-1:,0], points[-1:,1], points[-1:,2])
        
        # Update end effector trajectory path
        traj_points = np.array(end_effector_positions)
        traj_line.set_data_3d(traj_points[:,0], traj_points[:,1], traj_points[:,2])
        
        # Get current pose for label
        current_pose = forward_kinematics(q)
        end_pos = points[-1]
        rpy = current_pose[3:]
        
        # UPDATE SINGLE POSE LABEL - REMOVE OLD, CREATE NEW
        if pose_label is not None:
            pose_label.remove()
        
        # Two-line label: position on first line, orientation on second line
        label_text = f'({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})\n({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f})'
        pose_label = ax.text(end_pos[0], end_pos[1], end_pos[2]+0.05, 
                            label_text,
                            color='darkblue', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.95))
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(speed)
    
    # Hold the plot open for specified time after trajectory completion
    print(f"Trajectory complete. Displaying for {hold_time} seconds...")
    plt.pause(hold_time)
    
    plt.close(fig)



# Main simulation - FULL 6-DOF CONTROL
if __name__ == "__main__":
    # Extended arm home position
    q_current = np.array([1.2, 1.2, 0, 0, 0, 0])
    
    print("6-DOF Robot FULL POSE (Position + Orientation) Simulator")
    print("Commands: 'rel dx dy dz' or 'abs x y z R P Y' or 'quit'")
    print("Workspace: x=[-0.6,0.6], y=[-0.6,0.6], z=[0.2,1.0]")
    print("Orientation: R=Roll, P=Pitch, Y=Yaw (radians)")
    

    initial_pose = forward_kinematics(q_current)
    print(f"Initial pose:")
    print(f"  Position: [{initial_pose[0]:.3f}, {initial_pose[1]:.3f}, {initial_pose[2]:.3f}]")
    print(f"  Orientation (RPY): [{initial_pose[3]:.3f}, {initial_pose[4]:.3f}, {initial_pose[5]:.3f}] rad\n")
    
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
                if len(target_pose) == 3:
                    # Position-only command: maintain current orientation
                    current_pose = forward_kinematics(q_current)
                    target_pose = np.append(target_pose, current_pose[3:])
                print(f"Target Position: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
                print(f"Target Orientation (RPY): [{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}] rad")
           
            else:
                print("Invalid command!")
                continue
            
            print("Solving FULL 6-DOF IK (position-priority mode)...")
            q_target = inverse_kinematics(target_pose, q_current)
            
            if q_target is not None:
                trajectory = generate_trajectory(q_current, q_target, steps=50)
                print("Executing LIVE trajectory...")
                live_visualize_trajectory(trajectory, target_pose, speed=0.03, hold_time=5.0)
               
                q_current = q_target
                final_pose = forward_kinematics(q_current)
                pos_error = np.linalg.norm(final_pose[:3] - target_pose[:3])
                orient_error = orientation_error(final_pose[3:], target_pose[3:])
               
                print(f"\n✓ TARGET Position: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
                print(f"✓ ACTUAL Position: [{final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}]")
                print(f"  Position error: {pos_error*1000:.1f}mm")
                
                print(f"\n✓ TARGET Orientation (RPY): [{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}] rad")
                print(f"✓ ACTUAL Orientation (RPY): [{final_pose[3]:.3f}, {final_pose[4]:.3f}, {final_pose[5]:.3f}] rad")
                print(f"  Orientation error: [{orient_error[0]:.4f}, {orient_error[1]:.4f}, {orient_error[2]:.4f}] rad")
                print()
            else:
                print("✗ IK failed - target outside workspace or unreachable pose")
                print()
               
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            plt.close('all')
            sys.exit(0)
