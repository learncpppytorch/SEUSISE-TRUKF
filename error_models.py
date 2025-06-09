import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter
from scipy.ndimage import shift

class EnhancedErrorModels:
    @staticmethod
    def sins_error_model(true_att, true_vel, params):
        n = len(true_att)
        
        # 陀螺仪误差（固定偏置+随机游走+一阶马尔科夫）[7](@ref)
        gyro_bias = np.deg2rad(params.gyro_bias/3600)
        gyro_rw = np.deg2rad(params.gyro_rw/60)*np.sqrt(params.dt)*np.random.randn(n,3)
       
        # 使用一阶马尔科夫过程模拟陀螺仪漂移
        gyro_markov = np.zeros((n,3))
        for i in range(1,n):
            gyro_markov[i] = 0.9*gyro_markov[i-1] + 0.1*np.random.randn(3)
        gyro_err = gyro_bias + gyro_rw + 0.005*gyro_markov  # 减小马尔科夫过程的影响系数
        
        # 加速度计误差（固定偏置+马尔科夫过程）[3](@ref)
        accel_bias = params.accel_bias * 1e-3 * 9.8
        accel_markov = np.zeros((n,3))
        for i in range(1,n):
            accel_markov[i] = 0.95*accel_markov[i-1] + 0.05*np.random.randn(3)
        accel_err = accel_bias + 0.005*accel_markov
        
        return gyro_err, accel_err
    
    @staticmethod
    def dvl_measurement(true_vel, depth, t, params):
        """DVL测量模型（包含安装杆振动）"""
        n = true_vel.shape[1]

            # 生成三维振动角度（绕XYZ轴不同频率）
        vib_angle = np.array([
            # X轴振动：减小振动幅度，5Hz
            0.05 * np.sin(2 * np.pi * 5 * t),
            # Y轴振动：减小振动幅度，3Hz
            0.03 * np.sin(2 * np.pi * 3 * t + np.pi/4),
            # Z轴振动：减小振动幅度，2Hz
            0.04 * np.sin(2 * np.pi * 2 * t + np.pi/2)
        ])  # 单位：度
    
        # 转换振动角度到弧度并叠加安装偏差
        dynamic_angle = params.dvl_install_angle[:, None] + np.deg2rad(vib_angle)
    
        # 批量计算旋转矩阵
        vel_body = np.zeros_like(true_vel)
        for i in range(n):
        # 构建当前时刻的反对称矩阵
            so3_matrix = np.array([
                [0, -dynamic_angle[2,i], dynamic_angle[1,i]],
                [dynamic_angle[2,i], 0, -dynamic_angle[0,i]],
                [-dynamic_angle[1,i], dynamic_angle[0,i], 0]
            ])
            # 矩阵指数计算旋转矩阵
            R = expm(so3_matrix)
            vel_body[:,i] = R @ true_vel[:,i]
    
        # 底锁状态判断[9](@ref) 当航行器远离海底时，DVL无法获得底跟踪数据，垂向速度测量误差增大
        if np.mean(depth) > params.dvl_height_threshold:
            # 水层跟踪时垂向速度噪声增加（与深度成比例），减小噪声系数
            depth_factor = np.clip(depth / params.dvl_height_threshold, 1, 2)
            vel_body[2,:] += 0.05 * depth_factor * np.random.randn(n)
    
        # 复合误差模型
        scale_error = 1 + params.dvl_scale * np.random.randn(n)
        temp_effect = 1 + 0.001 * (t - t.mean())  # 温度漂移项
        return temp_effect * scale_error * vel_body + params.dvl_noise * np.random.randn(*vel_body.shape)

    @staticmethod
    def depth_measurement(true_depth, temp, params):
        # 非线性误差（二次项）
        nonlinear = params.depth_nonlinear * true_depth**2 * 1e-4
        # 温度漂移
        temp_effect = params.depth_temp_drift * (temp - 25)
        # 使用固定噪声标准差，与EKSF_OLD保持一致
        return true_depth + params.depth_bias + nonlinear + temp_effect + 0.01*np.random.randn()

    @staticmethod 
    def usbl_measurement(true_pos, t, params):
        """改进USBL（包含非整数时移插值处理）"""
        
        # 时间同步误差建模（支持非整数时移）
        time_shift = params.usbl_time_sync_error / params.dt
    
        # 三维位置插值处理（考虑各轴独立特性）
        shifted_pos = np.empty_like(true_pos)
        for i in range(3):  # 对X/Y/Z轴分别处理
            # 采用三次样条插值（mode='nearest'防止边界异常）
            shifted_pos[i] = shift(true_pos[i], 
                                shift=time_shift,
                                mode='nearest',
                                order=3)  # 三次样条插值
        
        # 仅对XY轴添加误差，Z轴保持真实值
        pos_error = np.zeros((3, len(t)))
        pos_error[0:2] = params.usbl_pos_error[0:2, None] * np.random.randn(1, len(t))
        
        # 仅对XY轴添加温度漂移
        temp_drift = np.zeros((3, len(t)))
        temp_drift[0:2] = 0.005 * (t - t.mean()) * np.random.randn(1, len(t))
        
        return shifted_pos + pos_error + temp_drift
