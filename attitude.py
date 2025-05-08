import numpy as np

def euler_to_quaternion(euler):
    """Convert Euler angles to quaternion
    Input: euler[3] - [roll, pitch, yaw]
    Output: quaternion[4] - [w, x, y, z]
    """
    roll, pitch, yaw = euler
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles
    Input: quaternion[4] - [w, x, y, z]
    Output: euler[3] - [roll, pitch, yaw]
    """
    w, x, y, z = q
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def quaternion_multiply(q1, q2):
    """Quaternion multiplication"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def attitude_update(current_att, gyro, dt):
    """Attitude update function
    Input:
        current_att: current attitude angles [roll, pitch, yaw]
        gyro: gyroscope measurements [wx, wy, wz]
        dt: time interval
    Output:
        new_att: updated attitude angles [roll, pitch, yaw]
    """
    # 1. Convert current attitude to quaternion
    current_quat = euler_to_quaternion(current_att)
    
    # 2. Convert angular velocity to quaternion increment
    wx, wy, wz = gyro
    dq = np.array([
        1,
        0.5 * wx * dt,
        0.5 * wy * dt,
        0.5 * wz * dt
    ])
    dq = dq / np.linalg.norm(dq)  # Normalize
    
    # 3. Update quaternion
    new_quat = quaternion_multiply(current_quat, dq)
    new_quat = new_quat / np.linalg.norm(new_quat)  # Normalize
    
    # 4. Convert back to Euler angles
    new_att = quaternion_to_euler(new_quat)
    
    return new_att

def gravity_compensation(accel, att):
    """Gravity compensation
    Input:
        accel: accelerometer measurements [ax, ay, az]
        att: attitude angles [roll, pitch, yaw]
    Output:
        compensated_accel: compensated acceleration
    """
    # Gravity acceleration
    g = 9.81
    
    # Calculate Direction Cosine Matrix (DCM)
    roll, pitch, yaw = att
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # DCM from navigation frame to body frame
    Cnb = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
        [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]
    ])
    
    # Gravity projection in navigation frame
    g_n = np.array([0, 0, g])
    
    # Remove gravity effect
    compensated_accel = accel - Cnb @ g_n
    
    return compensated_accel

def euler_to_rotation_matrix(euler):
    """Convert Euler angles to rotation matrix
    Input: euler[3] - [roll, pitch, yaw]
    Output: R[3,3] - rotation matrix from navigation frame to body frame
    """
    roll, pitch, yaw = euler
    
    # Calculate sine and cosine of the three angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Build rotation matrix
    R = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
        [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]
    ])
    
    return R

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles
    Input: R[3,3] - rotation matrix from navigation frame to body frame
    Output: euler[3] - [roll, pitch, yaw]
    """
    # Extract pitch angle (y-axis rotation)
    pitch = -np.arcsin(R[0,2])
    
    # Extract roll angle (x-axis rotation)
    roll = np.arctan2(R[1,2], R[2,2])
    
    # Extract yaw angle (z-axis rotation)
    yaw = np.arctan2(R[0,1], R[0,0])
    
    return np.array([roll, pitch, yaw])

def skew(v):
    """Calculate skew-symmetric matrix of a vector
    Input: v[3] - 3D vector
    Output: S[3,3] - skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])