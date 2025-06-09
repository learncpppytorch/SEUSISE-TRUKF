import numpy as np

class EnhancedSensorParams:
    def __init__(self):
        # SINS误差参数（增加时变特性）
        self.gyro_bias = np.array([0.1, 0.15, -0.08])  # 固定偏置(度/小时)
        self.gyro_rw = np.array([0.01, 0.015, 0.008])  # 随机游走(度/√h)
        self.accel_bias = np.array([0.02, 0.03, 0.01])  # 固定偏置(mg)
        self.accel_markov = np.array([0.005, 0.007, 0.003])  # 一阶马尔科夫过程(mg)
        self.init_att_error = np.deg2rad([0.5, 0.5, 1])  # 初始姿态误差
        
        # DVL误差模型
        self.dvl_scale = 5e-4  # 速度比例误差，减小一半
        self.dvl_noise = 0.05  # 白噪声(m/s)，减小一半
        self.dvl_install_angle = np.deg2rad([0.3, -0.2, 0.8])  # 安装偏差角[3](@ref)，减小安装偏差
        self.dvl_height_threshold = 150  # 底锁高度阈值(m)[9](@ref)
        
        # USBL误差模型
        self.usbl_pos_error = np.array([0.3, 0.3, 0.3])  # 位置误差标准差(m)，减小误差
        self.usbl_time_sync_error = 0.05  # 时间同步误差(s)[4](@ref)，减小时间同步误差
        self.usbl_update_interval = 5  # 秒
        
        # 深度传感器误差模型[6](@ref)
        self.depth_bias = 0.05  # 静态偏置(m)
        self.depth_nonlinear = 0.02 # 非线性误差(%FS)
        self.depth_temp_drift = 0.001  # 温度漂移(°C^-1)
        
        # 环境扰动参数[1](@ref)
        self.current_speed = 0.1  # 洋流速度(m/s)
        self.current_dir = np.deg2rad(45)  # 洋流方向
        self.wave_amplitude = 0.5  # 波浪扰动幅值(m)
        
        # 时间参数
        self.dt = 0.1  # 采样时间
        self.total_time = 600  # 总时长(秒)


        