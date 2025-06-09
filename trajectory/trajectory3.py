import numpy as np
from scipy.linalg import expm

def generate_true_trajectory(params):
    
    t = np.arange(0, params.total_time, params.dt) #生成离散时间
    n = len(t)
    
    # 初始化轨迹数组
    x = np.zeros(n)    
    y = np.zeros(n) 
    z = np.zeros(n) # 位置
    true_vel = np.zeros((3, n)) # 三维速度
    true_att = np.zeros((3, n))  # 姿态角[roll, pitch, yaw] (弧度)
    
    # 新阶段配置
    phases = [
        {   # 阶段1：缓速螺旋下潜（扩大螺旋半径）
            'type': 'spiral', 
            'duration': 120, # 持续时间
            'v_xy': 2.5,  # 水平速度
            'v_z': -0.5, # 垂直速度
            'omega': 2*np.pi/90,  # 角速度 (rad/s)
            'radius_growth': 0.6  # 螺旋半径增长系数
        },
        {   # 阶段2：斜向直行（覆盖Y轴负方向）
            'type': 'line',
            'duration': 120,
            'v_xy': 2.2,
            'v_z': -0.3,
            'direction': 315  # 315度方向航行
        },
        {   # 阶段3：三维蛇形机动（新增类型）
            'type': 'snake',
            'duration': 120,
            'v_xy': 1.8,
            'v_z': 0.2,
            'omega': 2*np.pi/60,  # 角速度 (rad/s)
            'amplitude': 120.0  # 摆动幅度\xy方向最大位移
        },
        {   # 阶段4：周期性起伏（Z轴正弦波动）
            'type': 'wave',
            'duration': 120,
            'v_xy': 0.5,
            'v_z': 0.6,
            'frequency': 2*np.pi/20,  # 波动周期
            'amplitude': 2.0
        },
        {   # 阶段5：大半径螺旋上升
            'type': 'spiral',
            'duration': 120,
            'v_xy': 2.0,
            'v_z': 0.5,
            'omega': 2*np.pi/90,  # 减缓旋转速度
            'radius_growth': 0.4  # 螺旋半径增长系数
        }
    ]
    
    current_idx = 0
    for phase in phases:
        phase_steps = int(phase['duration'] / params.dt) # 计算当前阶段步数
        phase_t = t[current_idx : current_idx+phase_steps] # 截取阶段时间
        
        if phase['type'] == 'spiral':
            # 螺旋（支持半径增长）
            r = phase.get('radius_growth', 0) * phase_t + phase['v_xy']
            x_p = r * np.cos(phase['omega']*phase_t)
            y_p = r * np.sin(phase['omega']*phase_t)
            z_p = phase['v_z'] * phase_t
            vx = -r * phase['omega'] * np.sin(phase['omega']*phase_t) + phase.get('radius_growth', 0) * np.cos(phase['omega']*phase_t)
            vy = r * phase['omega'] * np.cos(phase['omega']*phase_t) + phase.get('radius_growth', 0) * np.sin(phase['omega']*phase_t)
            vz = np.full_like(phase_t, phase['v_z'])
            yaw = phase['omega'] * phase_t  # 绕Z轴旋转
            true_att[2, current_idx:current_idx+phase_steps] = yaw

        elif phase['type'] == 'line':
            # 带Z轴运动的直线
            angle = np.deg2rad(phase['direction'])
            x_p = phase['v_xy'] * np.cos(angle) * phase_t
            y_p = phase['v_xy'] * np.sin(angle) * phase_t
            z_p = phase['v_z'] * phase_t
            vx = phase['v_xy'] * np.cos(angle) * np.ones_like(phase_t)
            vy = phase['v_xy'] * np.sin(angle) * np.ones_like(phase_t)
            vz = phase['v_z'] * np.ones_like(phase_t)
            # 修改部分：无条件初始化航向角
            init_yaw = np.deg2rad(phase['direction'])  # 始终使用当前阶段的方向角
            true_att[2, current_idx:current_idx+phase_steps] = init_yaw
            
        elif phase['type'] == 'snake':
            # 三维蛇形机动（X/Y交替摆动）
            x_p = phase['amplitude'] * np.sin(phase['omega']*phase_t)
            y_p = phase['amplitude'] * np.cos(phase['omega']*phase_t)
            z_p = phase['v_z'] * phase_t
            # 叠加前进分量
            x_p += phase['v_xy'] * 0.6 * phase_t
            y_p += phase['v_xy'] * 0.6 * phase_t
               # 速度计算（对x_p和y_p求导）
            vx = phase['amplitude'] * phase['omega'] * np.cos(phase['omega']*phase_t) + phase['v_xy'] * 0.6
            vy = -phase['amplitude'] * phase['omega'] * np.sin(phase['omega']*phase_t) + phase['v_xy'] * 0.6
            vz = np.full_like(phase_t, phase['v_z'])  # z方向速度恒定
            roll = 0.1 * np.sin(phase['omega']*phase_t)  # 绕X轴摆动
            true_att[0, current_idx:current_idx+phase_steps] = roll
        elif phase['type'] == 'wave':
            # 周期性起伏运动
            x_p = phase['v_xy'] * phase_t
            y_p = phase['v_xy'] * 0.5 * phase_t
            z_p = phase['amplitude'] * np.sin(phase['frequency']*phase_t)
                # 速度计算（对z_p求导）
            vx = np.full_like(phase_t, phase['v_xy'])          # x方向速度恒定
            vy = np.full_like(phase_t, phase['v_xy'] * 0.5)    # y方向速度恒定
            vz = phase['amplitude'] * phase['frequency'] * np.cos(phase['frequency']*phase_t)  # z方向速度波动
            pitch = 0.05 * np.sin(phase['frequency']*phase_t)  # 绕Y轴起伏
            true_att[1, current_idx:current_idx+phase_steps] = pitch
        # 轨迹叠加（保持连续性）
        start = current_idx
        end = current_idx + phase_steps
        x[start:end] = x_p + (x[current_idx-1] if current_idx>0 else 0)
        y[start:end] = y_p + (y[current_idx-1] if current_idx>0 else 0) 
        z[start:end] = z_p + (z[current_idx-1] if current_idx>0 else 0)
        #保存速度
        true_vel[0, start:end] = vx
        true_vel[1, start:end] = vy
        true_vel[2, start:end] = vz

        current_idx += phase_steps
        if current_idx >= n: break
    
    # 增强三维效果：添加XY平面扰动
    x += 2.5 * np.sin(2*np.pi*t/80)  # X轴波动[1](@ref)
    y += 2.5 * np.cos(2*np.pi*t/75)  # Y轴波动
    
    # 添加环境扰动

    current_vel = params.current_speed * np.array([
        np.cos(params.current_dir),
        np.sin(params.current_dir),
        0
    ])[:, None]
    true_vel += current_vel  # 叠加洋流速度
    
    # 添加波浪扰动（Z轴）
    wave = 0.2 * np.sin(2*np.pi*t/8)
    z += wave
    
    # 初始化数组
    true_gyro = np.zeros((3, n))
    true_accel = np.zeros((3, n))
    
    # 通过数值微分计算加速度
    true_accel[:, :-1] = np.diff(true_vel, axis=1) / params.dt
      
    true_gyro[:, 1:] = np.diff(true_att, axis=1)/params.dt  # 数值微分求角速度

    return np.vstack((x, y, z)), true_vel, true_gyro, true_accel, t
