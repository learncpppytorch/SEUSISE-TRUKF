import numpy as np

def generate_true_trajectory(params):
    """   
    包含典型的水下运动场景：下潜、平面巡航、定深航行、上浮等
    
    参数:
        params: 包含传感器和环境参数的对象
        
    返回:
        true_pos: 三维位置 [x,y,z]
        true_vel: 三维速度 [vx,vy,vz]
        true_gyro: 三维角速度 [wx,wy,wz]
        true_accel: 三维加速度 [ax,ay,az]
        t: 时间序列
    """
    # 生成离散时间序列
    t = np.arange(0, params.total_time, params.dt)
    n = len(t)
    
    # 初始化轨迹数组，设置初始位置为(0, 0, -0.01)
    x = np.full(n, -0.1)
    y = np.full(n, -0.1)
    z = np.full(n, -0.1)
    # true_vel = np.zeros((3, n))
    # true_att = np.zeros((3, n))  # 姿态角 [roll, pitch, yaw] (弧度)
    
    true_vel = np.full((3, n), -0.01)
    true_att = np.full((3, n), -0.01)
    # 定义简化的运动阶段
    phases = [
        # 阶段1：直线下潜，指向8字形的一个顶点
        {
            'type': 'dive',
            'duration': 60,  # 持续时间(秒)
            'v_xy': 0.35,     # 水平速度(m/s)
            'v_z': -0.1,     # 垂直速度(m/s)
            'direction': 270   # 水平航向(度)，指向8字形左上顶点
        },
        # 阶段2：定深8字形巡航
        {
            'type': 'figure8',
            'duration': 465,  # 持续时间(秒)
            'v_xy': 1.5,     # 水平速度(m/s)
            'amplitude': 30.0,  # 8字形幅度(m)
            'period': 375     # 完成一个8字形的周期(秒)
        },
        # 阶段3：缓慢上浮，从8字形的对称顶点出去
        {
            'type': 'ascend',
            'duration': 75,
            'v_xy': 0.3,
            'v_z': 0.1,
            'direction': 90   # 水平航向(度)，从8字形右下顶点出去
        }
    ]
    
    # 轨迹生成参数
    current_idx = 0
    last_pos = np.array([-0.1, -0.1, -0.1])  # 设置初始位置
    last_vel = np.zeros(3)  # 上一阶段结束速度
    last_yaw = 0            # 上一阶段结束航向角
    
    # 生成各阶段轨迹
    for i, phase in enumerate(phases):
        # 计算当前阶段步数
        phase_steps = int(phase['duration'] / params.dt)
        if current_idx + phase_steps > n:
            phase_steps = n - current_idx
        
        # 提取当前阶段时间片段
        phase_t = np.arange(0, phase_steps * params.dt, params.dt)
        
        # 初始化当前阶段轨迹
        x_p = np.zeros(phase_steps)
        y_p = np.zeros(phase_steps)
        z_p = np.zeros(phase_steps)
        vx = np.zeros(phase_steps)
        vy = np.zeros(phase_steps)
        vz = np.zeros(phase_steps)
        yaw = np.zeros(phase_steps)
        
        # 根据阶段类型生成轨迹
        if phase['type'] == 'dive':
            # 直线下潜
            angle = np.deg2rad(phase['direction'])
            
            # 平滑过渡到目标航向
            if i > 0:
                # 计算航向差，确保在-pi到pi之间
                angle_diff = (angle - last_yaw + np.pi) % (2 * np.pi) - np.pi
                # 创建平滑过渡函数（前10秒）
                trans_steps = min(int(10 / params.dt), phase_steps)
                trans_ratio = np.linspace(0, 1, trans_steps)
                
                for j in range(trans_steps):
                    yaw[j] = last_yaw + angle_diff * trans_ratio[j]
                yaw[trans_steps:] = angle
            else:
                yaw[:] = angle
            
            # 生成位置和速度
            for j in range(phase_steps):
                vx[j] = phase['v_xy'] * np.cos(yaw[j])
                vy[j] = phase['v_xy'] * np.sin(yaw[j])
                vz[j] = phase['v_z']
                
                # 积分得到位置
                if j > 0:
                    x_p[j] = x_p[j-1] + vx[j] * params.dt
                    y_p[j] = y_p[j-1] + vy[j] * params.dt
                    z_p[j] = z_p[j-1] + vz[j] * params.dt
        
        elif phase['type'] == 'figure8':
            # 定深8字形轨迹
            # 使用参数化方程生成8字形
            # 计算8字形的周期和角频率
            omega = 2 * np.pi / phase['period']  # 角频率
            
            # 生成8字形轨迹
            for j in range(phase_steps):
                # 参数方程：8字形 (使用莱姆尼斯卡特曲线的变种)
                t_param = omega * phase_t[j]
                # 使用参数化方程生成8字形轨迹
                x_p[j] = phase['amplitude'] * np.sin(t_param)
                y_p[j] = phase['amplitude'] * np.sin(t_param) * np.cos(t_param)
                z_p[j] = 0  # 定深航行
                
                # 计算速度（对位置求导）
                vx[j] = phase['amplitude'] * omega * np.cos(t_param)
                vy[j] = phase['amplitude'] * omega * (np.cos(t_param)**2 - np.sin(t_param)**2)
                vz[j] = 0  # 定深航行
                
                # 计算航向角（速度方向）
                yaw[j] = np.arctan2(vy[j], vx[j])
            
            # 调整速度大小，保持恒定的水平速度
            for j in range(phase_steps):
                # 计算当前水平速度大小
                v_horizontal = np.sqrt(vx[j]**2 + vy[j]**2)
                if v_horizontal > 1e-6:  # 避免除以零
                    # 调整速度大小为指定的水平速度
                    scale_factor = phase['v_xy'] / v_horizontal
                    vx[j] *= scale_factor
                    vy[j] *= scale_factor
            
            # 平滑过渡到8字形轨迹的初始航向
            if i > 0:
                # 创建平滑过渡函数（前10秒）
                trans_steps = min(int(10 / params.dt), phase_steps)
                trans_ratio = np.linspace(0, 1, trans_steps)
                
                # 保存原始计算的航向角和速度
                original_yaw = yaw.copy()
                original_vx = vx.copy()
                original_vy = vy.copy()
                
                # 平滑过渡航向角和速度
                for j in range(trans_steps):
                    # 平滑过渡航向角
                    angle_diff = (original_yaw[j] - last_yaw + np.pi) % (2 * np.pi) - np.pi
                    yaw[j] = last_yaw + angle_diff * trans_ratio[j]
                    
                    # 平滑过渡速度
                    vx[j] = last_vel[0] + (original_vx[j] - last_vel[0]) * trans_ratio[j]
                    vy[j] = last_vel[1] + (original_vy[j] - last_vel[1]) * trans_ratio[j]
                    
                    # 重新积分得到位置
                    if j > 0:
                        x_p[j] = x_p[j-1] + vx[j] * params.dt
                        y_p[j] = y_p[j-1] + vy[j] * params.dt
                        z_p[j] = z_p[j-1]
        
        elif phase['type'] == 'straight':
            # 定深直线航行
            angle = np.deg2rad(phase['direction'])
            
            # 平滑过渡到目标航向
            if i > 0:
                angle_diff = (angle - last_yaw + np.pi) % (2 * np.pi) - np.pi
                trans_steps = min(int(10 / params.dt), phase_steps)
                trans_ratio = np.linspace(0, 1, trans_steps)
                
                for j in range(trans_steps):
                    yaw[j] = last_yaw + angle_diff * trans_ratio[j]
                yaw[trans_steps:] = angle
            else:
                yaw[:] = angle
            
            # 生成位置和速度
            for j in range(phase_steps):
                vx[j] = phase['v_xy'] * np.cos(yaw[j])
                vy[j] = phase['v_xy'] * np.sin(yaw[j])
                vz[j] = 0  # 定深航行
                
                # 积分得到位置
                if j > 0:
                    x_p[j] = x_p[j-1] + vx[j] * params.dt
                    y_p[j] = y_p[j-1] + vy[j] * params.dt
                    z_p[j] = z_p[j-1]  # z保持不变
        
        elif phase['type'] == 'ascend':
            # 缓慢上浮
            angle = np.deg2rad(phase['direction'])
            
            # 平滑过渡到目标航向
            if i > 0:
                angle_diff = (angle - last_yaw + np.pi) % (2 * np.pi) - np.pi
                trans_steps = min(int(10 / params.dt), phase_steps)
                trans_ratio = np.linspace(0, 1, trans_steps)
                
                for j in range(trans_steps):
                    yaw[j] = last_yaw + angle_diff * trans_ratio[j]
                yaw[trans_steps:] = angle
            else:
                yaw[:] = angle
            
            # 生成位置和速度
            for j in range(phase_steps):
                vx[j] = phase['v_xy'] * np.cos(yaw[j])
                vy[j] = phase['v_xy'] * np.sin(yaw[j])
                vz[j] = phase['v_z']  # 上浮速度
                
                # 积分得到位置
                if j > 0:
                    x_p[j] = x_p[j-1] + vx[j] * params.dt
                    y_p[j] = y_p[j-1] + vy[j] * params.dt
                    z_p[j] = z_p[j-1] + vz[j] * params.dt
        
        # 调整轨迹以匹配上一阶段的终点
        if i > 0:
            x_p = x_p - x_p[0] + last_pos[0]
            y_p = y_p - y_p[0] + last_pos[1]
            z_p = z_p - z_p[0] + last_pos[2]
        
        # 保存轨迹
        start = current_idx
        end = current_idx + phase_steps
        x[start:end] = x_p
        y[start:end] = y_p
        z[start:end] = z_p
        
        # 保存速度
        true_vel[0, start:end] = vx
        true_vel[1, start:end] = vy
        true_vel[2, start:end] = vz
        
        # 保存姿态角
        true_att[2, start:end] = yaw  # 航向角
        
        # 添加俯仰角和横滚角（根据运动状态自然产生）
        if phase['type'] in ['dive', 'ascend']:
            # 下潜或上浮时产生俯仰角
            pitch_angle = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
            true_att[1, start:end] = pitch_angle
        
        # 记录当前阶段结束位置和速度，用于下一阶段的过渡
        if end < n:
            last_pos = np.array([x[end-1], y[end-1], z[end-1]])
            last_vel = np.array([true_vel[0, end-1], true_vel[1, end-1], true_vel[2, end-1]])
            last_yaw = true_att[2, end-1]
        
        current_idx += phase_steps
        if current_idx >= n:
            break
    
    # 添加环境扰动（洋流影响）
    current_vel = params.current_speed * np.array([
        np.cos(params.current_dir),
        np.sin(params.current_dir),
        0
    ])[:, None]
    
    # 叠加洋流速度
    true_vel += current_vel
    
    # 添加波浪扰动（Z轴）
    wave = 0.02 * np.sin(2*np.pi*t/8)
    z += wave
    
    # 初始化角速度和加速度数组
    true_gyro = np.zeros((3, n))
    true_accel = np.zeros((3, n))
    
    # 通过数值微分计算加速度和角速度
    true_accel[:, 1:] = np.diff(true_vel, axis=1) / params.dt
    true_gyro[:, 1:] = np.diff(true_att, axis=1) / params.dt
    
    # 返回结果
    return np.vstack((x, y, z)), true_vel, true_gyro, true_accel, t