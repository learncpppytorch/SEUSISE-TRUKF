import numpy as np

def euler_to_quaternion(euler):
    """欧拉角转四元数
    输入：euler[3] - [roll, pitch, yaw]
    输出：quaternion[4] - [w, x, y, z]
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
    """四元数转欧拉角
    输入：quaternion[4] - [w, x, y, z]
    输出：euler[3] - [roll, pitch, yaw]
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
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def attitude_update(current_att, gyro, dt):
    """姿态更新函数
    输入：
        current_att: 当前姿态角[roll, pitch, yaw]
        gyro: 陀螺仪测量值[wx, wy, wz]
        dt: 时间间隔
    输出：
        new_att: 更新后的姿态角[roll, pitch, yaw]
    """
    # 1. 当前姿态角转换为四元数
    current_quat = euler_to_quaternion(current_att)
    
    # 2. 角速度转换为四元数增量
    wx, wy, wz = gyro
    dq = np.array([
        1,
        0.5 * wx * dt,
        0.5 * wy * dt,
        0.5 * wz * dt
    ])
    dq = dq / np.linalg.norm(dq)  # 归一化
    
    # 3. 四元数更新
    new_quat = quaternion_multiply(current_quat, dq)
    new_quat = new_quat / np.linalg.norm(new_quat)  # 归一化
    
    # 4. 转换回欧拉角
    new_att = quaternion_to_euler(new_quat)
    
    return new_att

def gravity_compensation(accel, att):
    """重力补偿
    输入：
        accel: 加速度计测量值[ax, ay, az]
        att: 姿态角[roll, pitch, yaw]
    输出：
        compensated_accel: 补偿后的加速度
    """
    # 重力加速度
    g = 9.81
    
    # 计算姿态余弦矩阵（DCM）
    roll, pitch, yaw = att
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # DCM从导航系到机体系
    Cnb = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
        [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]
    ])
    
    # 重力在导航系下的投影
    g_n = np.array([0, 0, g])
    
    # 去除重力影响
    compensated_accel = accel - Cnb @ g_n
    
    return compensated_accel


    
def euler_to_rotation_matrix(euler):
    """欧拉角转旋转矩阵
    输入：euler[3] - [roll, pitch, yaw]
    输出：R[3,3] - 从导航系到机体系的旋转矩阵
    """
    roll, pitch, yaw = euler
    
    # 分别计算三个角度的正弦和余弦
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 构建旋转矩阵
    R = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
        [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]
    ])
    
    return R

def rotation_matrix_to_euler(R):
    """旋转矩阵转欧拉角
    输入：R[3,3] - 从导航系到机体系的旋转矩阵
    输出：euler[3] - [roll, pitch, yaw]
    """
    # 提取 pitch 角 (绕y轴)
    pitch = -np.arcsin(R[0,2])
    
    # 提取 roll 角 (绕x轴)
    roll = np.arctan2(R[1,2], R[2,2])
    
    # 提取 yaw 角 (绕z轴)
    yaw = np.arctan2(R[0,1], R[0,0])
    
    return np.array([roll, pitch, yaw])

def skew(v):
    """计算向量的反对称矩阵
    输入：v[3] - 三维向量
    输出：S[3,3] - 反对称矩阵
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])