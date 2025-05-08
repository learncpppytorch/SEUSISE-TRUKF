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
    
    # 用于保存各阶段起始位置和速度的变量，确保轨迹连续性
    phase_start_pos = np.zeros((3, 1))
    phase_start_vel = np.zeros((3, 1))
    
    # 新阶段配置
    phases = [
        {   # 阶段1：缓速螺旋下潜（扩大螺旋半径）
            'type': 'spiral', 
            'duration': 120, # 持续时间
            'v_xy': 1.5,  # 水平速度
            'v_z': -0.2, # 垂直速度
            'omega': 2*np.pi/90,  # 角速度 (rad/s)
            'radius_growth': 1.5  # 螺旋半径增长系数
        },
        {   # 阶段2：斜向直行（覆盖Y轴负方向）
            'type': 'line',
            'duration': 120,
            'v_xy': 2.5,
            'v_z': -0.1,
            'direction': 315  # 315度方向航行
        },
        {   # 阶段3：三维蛇形机动（新增类型）
            'type': 'snake',
            'duration': 120,
            'v_xy': 1.8,
            'v_z': 0,
            'omega': 2*np.pi/120,  # 角速度 (rad/s)
            'amplitude': 10.0  # 摆动幅度\xy方向最大位移
        },
        {   # 阶段4：周期性起伏（Z轴正弦波动）
            'type': 'wave',
            'duration': 120,
            'v_xy': 1.8,
            'v_z': 0.5,
            'frequency': 2*np.pi/60,  # 波动周期
            'amplitude': 1.0
        },
        {   # 阶段5：大半径螺旋上升
            'type': 'spiral',
            'duration': 120,
            'v_xy': 1.5,
            'v_z': 0.5,
            'omega': 2*np.pi/90,  # 减缓旋转速度
            'radius_growth': 1.5  # 螺旋半径增长系数
        }
    ]
    
    current_idx = 0
    phase_idx = 0
    for phase in phases:
        phase_steps = int(phase['duration'] / params.dt) # 计算当前阶段步数
        phase_t = t[current_idx : current_idx+phase_steps] # 截取阶段时间
        
        if phase['type'] == 'spiral':
            # 螺旋（支持半径增长）
            # 计算相对时间，使得每个阶段的时间从0开始
            relative_t = phase_t - phase_t[0]
            
            # 计算初始相位，使得速度方向与前一阶段连续
            initial_phase = 0
            if phase_idx > 0 and phase_start_vel[0, 0] != 0 and phase_start_vel[1, 0] != 0:
                initial_phase = np.arctan2(phase_start_vel[1, 0], phase_start_vel[0, 0])
                if phase['v_z'] * phase_start_vel[2, 0] < 0:  # 如果z方向速度变号，调整相位
                    initial_phase += np.pi
            
            # 计算初始半径，使得与前一阶段的位置连续
            initial_radius = phase['v_xy']
            if phase_idx > 0:
                # 计算到原点的距离作为初始半径
                xy_dist = np.sqrt(phase_start_pos[0, 0]**2 + phase_start_pos[1, 0]**2)
                if xy_dist > 0:
                    initial_radius = max(xy_dist, phase['v_xy'])
            
            # 生成螺旋轨迹
            r = phase.get('radius_growth', 0) * relative_t + initial_radius
            x_p = r * np.cos(phase['omega']*relative_t + initial_phase)
            y_p = r * np.sin(phase['omega']*relative_t + initial_phase)
            z_p = phase['v_z'] * relative_t
            
            # 调整初始位置以匹配前一阶段的终点
            if phase_idx > 0:
                x_p = x_p - x_p[0] + phase_start_pos[0, 0]
                y_p = y_p - y_p[0] + phase_start_pos[1, 0]
                z_p = z_p - z_p[0] + phase_start_pos[2, 0]
            
            # 计算速度
            vx = -r * phase['omega'] * np.sin(phase['omega']*relative_t + initial_phase) + phase.get('radius_growth', 0) * np.cos(phase['omega']*relative_t + initial_phase)
            vy = r * phase['omega'] * np.cos(phase['omega']*relative_t + initial_phase) + phase.get('radius_growth', 0) * np.sin(phase['omega']*relative_t + initial_phase)
            vz = np.full_like(relative_t, phase['v_z'])
            
            # 平滑速度过渡
            if phase_idx > 0 and len(relative_t) > 0:
                # 创建平滑过渡函数（前10个时间步）
                transition_steps = min(10, len(relative_t))
                transition = np.linspace(0, 1, transition_steps)
                
                # 应用速度平滑过渡
                for i in range(transition_steps):
                    blend = transition[i]
                    vx[i] = (1-blend) * phase_start_vel[0, 0] + blend * vx[i]
                    vy[i] = (1-blend) * phase_start_vel[1, 0] + blend * vy[i]
                    vz[i] = (1-blend) * phase_start_vel[2, 0] + blend * vz[i]
            
            # 更新姿态
            roll = 0.1 * np.sin(phase['omega']*relative_t + initial_phase)  # 绕X轴摆动
            true_att[0, current_idx:current_idx+phase_steps] = roll
            
            # 更新姿态
            yaw = phase['omega'] * relative_t + initial_phase  # 绕Z轴旋转
            true_att[2, current_idx:current_idx+phase_steps] = yaw

        elif phase['type'] == 'line':
            # 带Z轴运动的直线
            # 计算相对时间，使得每个阶段的时间从0开始
            relative_t = phase_t - phase_t[0]
            
            # 计算航向角，如果是第一个阶段，使用指定方向；否则尝试匹配前一阶段的速度方向
            angle = np.deg2rad(phase['direction'])
            if phase_idx > 0 and (np.abs(phase_start_vel[0, 0]) > 1e-6 or np.abs(phase_start_vel[1, 0]) > 1e-6):
                prev_angle = np.arctan2(phase_start_vel[1, 0], phase_start_vel[0, 0])
                # 平滑过渡到目标角度
                transition_steps = min(10, len(relative_t))
                angle_diff = (angle - prev_angle + np.pi) % (2 * np.pi) - np.pi  # 确保差值在-pi到pi之间
                angles = np.ones_like(relative_t) * angle
                for i in range(transition_steps):
                    blend = i / transition_steps
                    angles[i] = prev_angle + blend * angle_diff
            else:
                angles = np.ones_like(relative_t) * angle
            
            # 生成直线轨迹
            x_p = np.zeros_like(relative_t)
            y_p = np.zeros_like(relative_t)
            for i in range(len(relative_t)):
                x_p[i] = phase['v_xy'] * np.cos(angles[i]) * relative_t[i]
                y_p[i] = phase['v_xy'] * np.sin(angles[i]) * relative_t[i]
            z_p = phase['v_z'] * relative_t
            
            # 调整初始位置以匹配前一阶段的终点
            if phase_idx > 0:
                x_p = x_p - x_p[0] + phase_start_pos[0, 0]
                y_p = y_p - y_p[0] + phase_start_pos[1, 0]
                z_p = z_p - z_p[0] + phase_start_pos[2, 0]
            
            # 计算速度
            vx = np.zeros_like(relative_t)
            vy = np.zeros_like(relative_t)
            for i in range(len(relative_t)):
                vx[i] = phase['v_xy'] * np.cos(angles[i])
                vy[i] = phase['v_xy'] * np.sin(angles[i])
            vz = np.full_like(relative_t, phase['v_z'])
            
            # 平滑速度过渡
            if phase_idx > 0 and len(relative_t) > 0:
                # 创建平滑过渡函数（前10个时间步）
                transition_steps = min(10, len(relative_t))
                transition = np.linspace(0, 1, transition_steps)
                
                # 应用速度平滑过渡
                for i in range(transition_steps):
                    blend = transition[i]
                    vx[i] = (1-blend) * phase_start_vel[0, 0] + blend * vx[i]
                    vy[i] = (1-blend) * phase_start_vel[1, 0] + blend * vy[i]
                    vz[i] = (1-blend) * phase_start_vel[2, 0] + blend * vz[i]
            
            # 更新姿态
            roll = 0.1 * np.sin(phase['direction']*relative_t + initial_phase)  # 绕X轴摆动
            true_att[0, current_idx:current_idx+phase_steps] = roll
            
            # 更新姿态
            true_att[2, current_idx:current_idx+phase_steps] = angles
            
        elif phase['type'] == 'snake':
            # 三维蛇形机动（X/Y交替摆动）
            # 计算相对时间，使得每个阶段的时间从0开始
            relative_t = phase_t - phase_t[0]
            
            # 计算初始相位以匹配前一阶段的位置和速度
            initial_phase = 0
            if phase_idx > 0:
                # 尝试匹配前一阶段的速度方向
                prev_vx = phase_start_vel[0, 0]
                prev_vy = phase_start_vel[1, 0]
                if np.abs(prev_vx) > 1e-6 or np.abs(prev_vy) > 1e-6:
                    initial_phase = np.arctan2(prev_vy, prev_vx)
            
            # 计算基本轨迹
            x_p = phase['amplitude'] * np.sin(phase['omega']*phase_t + initial_phase)
            y_p = phase['amplitude'] * np.cos(phase['omega']*phase_t + initial_phase)
            z_p = phase['v_z'] * phase_t
            
            # 叠加前进分量
            x_p += phase['v_xy'] * 0.6 * phase_t
            y_p += phase['v_xy'] * 0.6 * phase_t
            
            # 调整初始位置以匹配前一阶段的终点
            x_p = x_p - x_p[0] + phase_start_pos[0, 0]
            y_p = y_p - y_p[0] + phase_start_pos[1, 0]
            z_p = z_p - z_p[0] + phase_start_pos[2, 0]
            
            # 速度计算（对x_p和y_p求导）
            vx = phase['amplitude'] * phase['omega'] * np.cos(phase['omega']*phase_t + initial_phase) + phase['v_xy'] * 0.6
            vy = -phase['amplitude'] * phase['omega'] * np.sin(phase['omega']*phase_t + initial_phase) + phase['v_xy'] * 0.6
            vz = np.full_like(phase_t, phase['v_z'])  # z方向速度恒定
            
            # 平滑速度过渡
            if phase_idx > 0 and len(relative_t) > 0:
                # 创建平滑过渡函数（前10个时间步）
                transition_steps = min(10, len(relative_t))
                transition = np.linspace(0, 1, transition_steps)
                
                # 应用速度平滑过渡
                for i in range(transition_steps):
                    blend = transition[i]
                    vx[i] = (1-blend) * phase_start_vel[0, 0] + blend * vx[i]
                    vy[i] = (1-blend) * phase_start_vel[1, 0] + blend * vy[i]
                    vz[i] = (1-blend) * phase_start_vel[2, 0] + blend * vz[i]
            
            # 更新姿态
            roll = 0.1 * np.sin(phase['omega']*relative_t + initial_phase)  # 绕X轴摆动
            true_att[0, current_idx:current_idx+phase_steps] = roll
        elif phase['type'] == 'wave':
            # 周期性起伏运动
            # 计算相对时间，使得每个阶段的时间从0开始
            relative_t = phase_t - phase_t[0]
            
            # 计算初始相位，使得z方向速度与前一阶段连续
            initial_phase = 0
            if phase_idx > 0 and np.abs(phase_start_vel[2, 0]) > 1e-6:
                # 根据前一阶段的z方向速度计算初始相位
                # 如果前一阶段z速度为0，则初始相位为0（波峰）
                # 如果前一阶段z速度为正，则初始相位为-pi/2（上升段）
                # 如果前一阶段z速度为负，则初始相位为pi/2（下降段）
                if phase_start_vel[2, 0] > 0:
                    initial_phase = -np.pi/2
                elif phase_start_vel[2, 0] < 0:
                    initial_phase = np.pi/2
            
            # 生成基本轨迹
            x_p = phase['v_xy'] * relative_t
            y_p = phase['v_xy'] * 0.5 * relative_t
            z_p = phase['amplitude'] * np.sin(phase['frequency']*relative_t + initial_phase)
            
            # 调整初始位置以匹配前一阶段的终点
            if phase_idx > 0:
                x_p = x_p - x_p[0] + phase_start_pos[0, 0]
                y_p = y_p - y_p[0] + phase_start_pos[1, 0]
                z_p = z_p - z_p[0] + phase_start_pos[2, 0]
            
            # 速度计算
            vx = np.full_like(relative_t, phase['v_xy'])          # x方向速度恒定
            vy = np.full_like(relative_t, phase['v_xy'] * 0.5)    # y方向速度恒定
            vz = phase['amplitude'] * phase['frequency'] * np.cos(phase['frequency']*relative_t + initial_phase)  # z方向速度波动
            
            # 平滑速度过渡
            if phase_idx > 0 and len(relative_t) > 0:
                # 创建平滑过渡函数（前10个时间步）
                transition_steps = min(10, len(relative_t))
                transition = np.linspace(0, 1, transition_steps)
                
                # 应用速度平滑过渡
                for i in range(transition_steps):
                    blend = transition[i]
                    vx[i] = (1-blend) * phase_start_vel[0, 0] + blend * vx[i]
                    vy[i] = (1-blend) * phase_start_vel[1, 0] + blend * vy[i]
                    vz[i] = (1-blend) * phase_start_vel[2, 0] + blend * vz[i]
            
            # 更新姿态
            roll = 0.1 * np.sin(phase['frequency']*relative_t + initial_phase)  # 绕X轴摆动
            true_att[0, current_idx:current_idx+phase_steps] = roll
            
            # 更新姿态
            pitch = 0.05 * np.sin(phase['frequency']*relative_t + initial_phase)  # 绕Y轴起伏
            true_att[1, current_idx:current_idx+phase_steps] = pitch
        # 直接使用计算好的轨迹（已经处理了连续性）
        start = current_idx
        end = current_idx + phase_steps
        x[start:end] = x_p
        y[start:end] = y_p
        z[start:end] = z_p
        
        # 保存速度
        true_vel[0, start:end] = vx
        true_vel[1, start:end] = vy
        true_vel[2, start:end] = vz

        # 保存当前阶段的终点位置和速度，用于下一阶段的初始化
        if current_idx + phase_steps < n:
            phase_end_idx = current_idx + phase_steps - 1
            phase_start_pos = np.array([[x[phase_end_idx]], 
                                       [y[phase_end_idx]], 
                                       [z[phase_end_idx]]])
            phase_start_vel = np.array([[true_vel[0, phase_end_idx]], 
                                       [true_vel[1, phase_end_idx]], 
                                       [true_vel[2, phase_end_idx]]])
        
        current_idx += phase_steps
        phase_idx += 1
        if current_idx >= n: break
    
    # 增强三维效果：添加XY平面扰动（使用平滑函数确保连续性）
    xy_disturbance_x = 2.5 * np.sin(2*np.pi*t/80)  # X轴波动
    xy_disturbance_y = 2.5 * np.cos(2*np.pi*t/75)  # Y轴波动
    
    # 计算扰动的导数（速度影响）
    xy_vel_x = 2.5 * (2*np.pi/80) * np.cos(2*np.pi*t/80)
    xy_vel_y = -2.5 * (2*np.pi/75) * np.sin(2*np.pi*t/75)
    
    # 添加扰动到位置
    x += xy_disturbance_x
    y += xy_disturbance_y
    
    # 添加环境扰动（洋流）
    current_vel = params.current_speed * np.array([
        np.cos(params.current_dir),
        np.sin(params.current_dir),
        0
    ])[:, None]
    
    # 添加波浪扰动（Z轴）
    wave_amplitude = 0.2
    wave_period = 8
    wave = wave_amplitude * np.sin(2*np.pi*t/wave_period)
    wave_vel = wave_amplitude * (2*np.pi/wave_period) * np.cos(2*np.pi*t/wave_period)
    
    # 更新位置和速度
    z += wave
    
    # 更新速度（包括所有扰动的影响）
    true_vel[0, :] += xy_vel_x + current_vel[0, 0]
    true_vel[1, :] += xy_vel_y + current_vel[1, 0]
    true_vel[2, :] += wave_vel + current_vel[2, 0]
    
    # 初始化数组
    true_gyro = np.zeros((3, n))
    true_accel = np.zeros((3, n))
    
    # 通过数值微分计算加速度
    true_accel[:, :-1] = np.diff(true_vel, axis=1) / params.dt
      
    true_gyro[:, 1:] = np.diff(true_att, axis=1)/params.dt  # 数值微分求角速度

    return np.vstack((x, y, z)), true_vel, true_gyro, true_accel, t