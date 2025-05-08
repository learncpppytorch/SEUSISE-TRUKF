import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import time
from scipy import stats
from scipy.signal import correlate

# 导入自定义模块
from params import EnhancedSensorParams
from trajectory.trajectory import generate_true_trajectory
from error_models import EnhancedErrorModels
from filters.ekf import EKF
from filters.eskf import ESKF
from filters.ukf import UKF
from filters.trukf import TRUKF
def run_monte_carlo_simulation(num_simulations=1000):
    """
    运行EKF、ESKF、UKF和TRUKF滤波器的蒙特卡洛仿真
    
    参数:
        num_simulations: 要运行的蒙特卡洛仿真次数
        
    返回:
        包含仿真结果的字典
    """
    # 初始化参数
    params = EnhancedSensorParams()
    
    # 如果结果目录不存在则创建
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 初始化数组以存储结果
    # 我们将存储每次仿真的最终位置误差
    ekf_errors = np.zeros((num_simulations, 3))
    eskf_errors = np.zeros((num_simulations, 3))
    ukf_errors = np.zeros((num_simulations, 3))
    trukf_errors = np.zeros((num_simulations, 3))
    
    # 用于存储每次仿真的均方根误差(RMSE)的数组
    ekf_rmse = np.zeros(num_simulations)
    eskf_rmse = np.zeros(num_simulations)
    ukf_rmse = np.zeros(num_simulations)
    trukf_rmse = np.zeros(num_simulations)
    
    # 用于存储NEES(归一化估计误差平方)的数组
    ekf_nees = []
    eskf_nees = []
    ukf_nees = []
    trukf_nees = []
    
    # 用于存储单步计算时延的数组
    ekf_time = []
    eskf_time = []
    ukf_time = []
    trukf_time = []
    
    # 用于存储归一化新息平方(NIS)的数组
    ekf_nis = []
    eskf_nis = []
    ukf_nis = []
    trukf_nis = []
    
    # 用于存储自相关函数的数组
    ekf_autocorr = []
    eskf_autocorr = []
    ukf_autocorr = []
    trukf_autocorr = []
    
    # 用于存储噪声敏感系数的数组
    ekf_noise_sensitivity = np.zeros(num_simulations)
    eskf_noise_sensitivity = np.zeros(num_simulations)
    ukf_noise_sensitivity = np.zeros(num_simulations)
    trukf_noise_sensitivity = np.zeros(num_simulations)
    
    # 对于第一次仿真，我们将保存完整的轨迹数据用于可视化
    save_trajectory = True
    
    # 运行蒙特卡洛仿真
    for sim_idx in tqdm(range(num_simulations), desc="正在运行蒙特卡洛仿真"):
        # 生成真实轨迹
        true_pos, true_vel, true_gyro, true_accel, t = generate_true_trajectory(params)
        n = len(t)
        
        # 生成带有随机误差的传感器数据
        gyro_err, accel_err = EnhancedErrorModels.sins_error_model(
            np.zeros((n, 3)), true_vel, params)
        
        depth_meas = EnhancedErrorModels.depth_measurement(
            true_pos[2], 10 + 0.1 * np.random.randn(n), params)
        dvl_meas = EnhancedErrorModels.dvl_measurement(
            true_vel, true_pos[2], t, params)
        usbl_meas = EnhancedErrorModels.usbl_measurement(true_pos, t, params)
        
        # 初始化滤波器
        ekf = EKF(params)
        eskf = ESKF(params)
        ukf = UKF(params)
        trukf = TRUKF(params)
        
        # 初始化状态估计
        ekf.pos = true_pos[:, 0].copy()
        ekf.vel = true_vel[:, 0].copy()
        
        eskf.pos_nominal = true_pos[:, 0].copy()
        eskf.vel_nominal = true_vel[:, 0].copy()
        
        # 初始化本次仿真的存储数组
        ekf_pos = np.zeros_like(true_pos)
        eskf_pos = np.zeros_like(true_pos)
        ukf_pos = np.zeros_like(true_pos)
        trukf_pos = np.zeros_like(true_pos)
        
        ekf_pos[:, 0] = ekf.pos
        eskf_pos[:, 0] = eskf.pos_nominal
        ukf_pos[:, 0] = ukf.pos_nominal
        trukf_pos[:, 0] = trukf.get_position()
        
        # 初始化本次仿真的NEES和NIS数组
        ekf_nees_sim = []
        eskf_nees_sim = []
        ukf_nees_sim = []
        trukf_nees_sim = []
        
        ekf_nis_sim = []
        eskf_nis_sim = []
        ukf_nis_sim = []
        trukf_nis_sim = []
        
        # 初始化本次仿真的误差序列（用于计算自相关函数）
        ekf_err_seq = np.zeros((3, n))
        eskf_err_seq = np.zeros((3, n))
        ukf_err_seq = np.zeros((3, n))
        trukf_err_seq = np.zeros((3, n))
        
        # 主仿真循环
        for i in range(1, n):
            # 获取IMU测量值
            gyro_meas = true_gyro[:, i] + gyro_err[i]
            accel_meas = true_accel[:, i] + accel_err[i]
            
            # EKF预测和更新
            start_time = time.time()
            ekf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz USBL更新
                nis = ekf.update(usbl_meas[:, i], 'USBL', return_nis=True)
                if nis is not None:
                    ekf_nis_sim.append(nis)
            if i % int(0.1/params.dt) == 0:  # 10Hz DVL和深度更新
                nis_dvl = ekf.update(dvl_meas[:, i], 'DVL', return_nis=True)
                if nis_dvl is not None:
                    ekf_nis_sim.append(nis_dvl)
                nis_depth = ekf.update(np.array([depth_meas[i]]), 'Depth', return_nis=True)
                if nis_depth is not None:
                    ekf_nis_sim.append(nis_depth)
            ekf_time.append(time.time() - start_time)
            ekf_pos[:, i] = ekf.pos.copy()
            
            # 计算NEES（如果滤波器提供协方差矩阵）
            if hasattr(ekf, 'P'):
                err = true_pos[:, i] - ekf.pos
                nees = err.T @ np.linalg.inv(ekf.P[:3, :3]) @ err
                ekf_nees_sim.append(nees)
            
            # 记录误差序列
            ekf_err_seq[:, i] = true_pos[:, i] - ekf_pos[:, i]
            
            # ESKF预测和更新
            start_time = time.time()
            eskf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz USBL更新
                nis = eskf.update(usbl_meas[:, i], 'USBL', return_nis=True)
                if nis is not None:
                    eskf_nis_sim.append(nis)
            if i % int(0.1/params.dt) == 0:  # 10Hz DVL和深度更新
                nis_dvl = eskf.update(dvl_meas[:, i], 'DVL', return_nis=True)
                if nis_dvl is not None:
                    eskf_nis_sim.append(nis_dvl)
                nis_depth = eskf.update(np.array([depth_meas[i]]), 'Depth', return_nis=True)
                if nis_depth is not None:
                    eskf_nis_sim.append(nis_depth)
            eskf_time.append(time.time() - start_time)
            eskf_pos[:, i] = eskf.pos_nominal.copy()
            
            # 计算NEES
            if hasattr(eskf, 'P'):
                err = true_pos[:, i] - eskf.pos_nominal
                nees = err.T @ np.linalg.inv(eskf.P[:3, :3]) @ err
                eskf_nees_sim.append(nees)
            
            # 记录误差序列
            eskf_err_seq[:, i] = true_pos[:, i] - eskf_pos[:, i]
            
            # UKF预测和更新
            start_time = time.time()
            ukf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz更新
                if not np.isnan(depth_meas[i]):
                    nis = ukf.update(np.array([depth_meas[i]]), 'Depth', i, return_nis=True)
                    if nis is not None:
                        ukf_nis_sim.append(nis)
                if not np.any(np.isnan(usbl_meas[:, i])):
                    nis = ukf.update(usbl_meas[:, i], 'USBL', i, return_nis=True)
                    if nis is not None:
                        ukf_nis_sim.append(nis)
                if not np.any(np.isnan(dvl_meas[:, i])):
                    nis = ukf.update(dvl_meas[:, i], 'DVL', i, return_nis=True)
                    if nis is not None:
                        ukf_nis_sim.append(nis)
            ukf_time.append(time.time() - start_time)
            ukf_pos[:, i] = ukf.pos_nominal.copy()
            
            # 计算NEES
            if hasattr(ukf, 'P'):
                err = true_pos[:, i] - ukf.pos_nominal
                nees = err.T @ np.linalg.inv(ukf.P[:3, :3]) @ err
                ukf_nees_sim.append(nees)
            
            # 记录误差序列
            ukf_err_seq[:, i] = true_pos[:, i] - ukf_pos[:, i]
            
            # TRUKF预测和更新
            start_time = time.time()
            trukf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz更新
                if not np.isnan(depth_meas[i]):
                    nis = trukf.update(np.array([depth_meas[i]]), 'Depth', i, return_nis=True)
                    if nis is not None:
                        trukf_nis_sim.append(nis)
                if not np.any(np.isnan(usbl_meas[:, i])):
                    nis = trukf.update(usbl_meas[:, i], 'USBL', i, return_nis=True)
                    if nis is not None:
                        trukf_nis_sim.append(nis)
                if not np.any(np.isnan(dvl_meas[:, i])):
                    nis = trukf.update(dvl_meas[:, i], 'DVL', i, return_nis=True)
                    if nis is not None:
                        trukf_nis_sim.append(nis)
            trukf_time.append(time.time() - start_time)
            trukf_pos[:, i] = trukf.get_position().copy()
            
            # 计算NEES
            if hasattr(trukf, 'P'):
                err = true_pos[:, i] - trukf.get_position()
                nees = err.T @ np.linalg.inv(trukf.P[:3, :3]) @ err
                trukf_nees_sim.append(nees)
            
            # 记录误差序列
            trukf_err_seq[:, i] = true_pos[:, i] - trukf_pos[:, i]
        
        # 计算最终位置误差
        ekf_errors[sim_idx] = true_pos[:, -1] - ekf_pos[:, -1]
        eskf_errors[sim_idx] = true_pos[:, -1] - eskf_pos[:, -1]
        ukf_errors[sim_idx] = true_pos[:, -1] - ukf_pos[:, -1]
        trukf_errors[sim_idx] = true_pos[:, -1] - trukf_pos[:, -1]
        
        # 计算每个滤波器的RMSE
        ekf_rmse[sim_idx] = np.sqrt(np.mean(np.sum((true_pos - ekf_pos)**2, axis=0)))
        eskf_rmse[sim_idx] = np.sqrt(np.mean(np.sum((true_pos - eskf_pos)**2, axis=0)))
        ukf_rmse[sim_idx] = np.sqrt(np.mean(np.sum((true_pos - ukf_pos)**2, axis=0)))
        trukf_rmse[sim_idx] = np.sqrt(np.mean(np.sum((true_pos - trukf_pos)**2, axis=0)))
        
        # 计算噪声敏感系数（误差与输入噪声的比值）
        gyro_noise_power = np.mean(np.sum(gyro_err**2, axis=1))
        accel_noise_power = np.mean(np.sum(accel_err**2, axis=1))
        total_noise_power = gyro_noise_power + accel_noise_power
        
        ekf_noise_sensitivity[sim_idx] = ekf_rmse[sim_idx] / total_noise_power if total_noise_power > 0 else 0
        eskf_noise_sensitivity[sim_idx] = eskf_rmse[sim_idx] / total_noise_power if total_noise_power > 0 else 0
        ukf_noise_sensitivity[sim_idx] = ukf_rmse[sim_idx] / total_noise_power if total_noise_power > 0 else 0
        trukf_noise_sensitivity[sim_idx] = trukf_rmse[sim_idx] / total_noise_power if total_noise_power > 0 else 0
        
        # 计算自相关函数
        max_lag = min(50, n-1)  # 最大滞后值
        
        # 对每个维度计算自相关函数
        ekf_autocorr_sim = []
        eskf_autocorr_sim = []
        ukf_autocorr_sim = []
        trukf_autocorr_sim = []
        
        for dim in range(3):
            # 归一化误差序列
            ekf_err_norm = (ekf_err_seq[dim, 1:] - np.mean(ekf_err_seq[dim, 1:])) / np.std(ekf_err_seq[dim, 1:]) if np.std(ekf_err_seq[dim, 1:]) > 0 else np.zeros(n-1)
            eskf_err_norm = (eskf_err_seq[dim, 1:] - np.mean(eskf_err_seq[dim, 1:])) / np.std(eskf_err_seq[dim, 1:]) if np.std(eskf_err_seq[dim, 1:]) > 0 else np.zeros(n-1)
            ukf_err_norm = (ukf_err_seq[dim, 1:] - np.mean(ukf_err_seq[dim, 1:])) / np.std(ukf_err_seq[dim, 1:]) if np.std(ukf_err_seq[dim, 1:]) > 0 else np.zeros(n-1)
            trukf_err_norm = (trukf_err_seq[dim, 1:] - np.mean(trukf_err_seq[dim, 1:])) / np.std(trukf_err_seq[dim, 1:]) if np.std(trukf_err_seq[dim, 1:]) > 0 else np.zeros(n-1)
            
            # 计算自相关函数
            ekf_acorr = correlate(ekf_err_norm, ekf_err_norm, mode='full')[len(ekf_err_norm)-1:len(ekf_err_norm)+max_lag] / len(ekf_err_norm)
            eskf_acorr = correlate(eskf_err_norm, eskf_err_norm, mode='full')[len(eskf_err_norm)-1:len(eskf_err_norm)+max_lag] / len(eskf_err_norm)
            ukf_acorr = correlate(ukf_err_norm, ukf_err_norm, mode='full')[len(ukf_err_norm)-1:len(ukf_err_norm)+max_lag] / len(ukf_err_norm)
            trukf_acorr = correlate(trukf_err_norm, trukf_err_norm, mode='full')[len(trukf_err_norm)-1:len(trukf_err_norm)+max_lag] / len(trukf_err_norm)
            
            ekf_autocorr_sim.append(ekf_acorr)
            eskf_autocorr_sim.append(eskf_acorr)
            ukf_autocorr_sim.append(ukf_acorr)
            trukf_autocorr_sim.append(trukf_acorr)
        
        # 存储NEES、NIS和自相关函数
        if ekf_nees_sim:
            ekf_nees.append(ekf_nees_sim)
        if eskf_nees_sim:
            eskf_nees.append(eskf_nees_sim)
        if ukf_nees_sim:
            ukf_nees.append(ukf_nees_sim)
        if trukf_nees_sim:
            trukf_nees.append(trukf_nees_sim)
        
        if ekf_nis_sim:
            ekf_nis.append(ekf_nis_sim)
        if eskf_nis_sim:
            eskf_nis.append(eskf_nis_sim)
        if ukf_nis_sim:
            ukf_nis.append(ukf_nis_sim)
        if trukf_nis_sim:
            trukf_nis.append(trukf_nis_sim)
        
        ekf_autocorr.append(ekf_autocorr_sim)
        eskf_autocorr.append(eskf_autocorr_sim)
        ukf_autocorr.append(ukf_autocorr_sim)
        trukf_autocorr.append(trukf_autocorr_sim)
        
        # 保存第一次仿真的轨迹数据
        if save_trajectory:
            # 创建包含轨迹数据的DataFrame
            trajectory_data = {
                'Time': t,
                'True_X': true_pos[0],
                'True_Y': true_pos[1],
                'True_Z': true_pos[2],
                'EKF_X': ekf_pos[0],
                'EKF_Y': ekf_pos[1],
                'EKF_Z': ekf_pos[2],
                'ESKF_X': eskf_pos[0],
                'ESKF_Y': eskf_pos[1],
                'ESKF_Z': eskf_pos[2],
                'UKF_X': ukf_pos[0],
                'UKF_Y': ukf_pos[1],
                'UKF_Z': ukf_pos[2],
                'TRUKF_X': trukf_pos[0],
                'TRUKF_Y': trukf_pos[1],
                'TRUKF_Z': trukf_pos[2]
            }
            df_trajectory = pd.DataFrame(trajectory_data)
            df_trajectory.to_excel(os.path.join(results_dir, 'monte_carlo_trajectories_0.5.xlsx'), index=False)
            save_trajectory = False
    
    # 计算误差的均值和标准差
    ekf_mean_error = np.mean(np.abs(ekf_errors), axis=0)
    eskf_mean_error = np.mean(np.abs(eskf_errors), axis=0)
    ukf_mean_error = np.mean(np.abs(ukf_errors), axis=0)  # 修正了这里的错误，之前是eskf_errors
    trukf_mean_error = np.mean(np.abs(trukf_errors), axis=0)
    
    ekf_std_error = np.std(np.abs(ekf_errors), axis=0)
    eskf_std_error = np.std(np.abs(eskf_errors), axis=0)
    ukf_std_error = np.std(np.abs(ukf_errors), axis=0)
    trukf_std_error = np.std(np.abs(trukf_errors), axis=0)
    
    # 计算平均RMSE
    ekf_mean_rmse = np.mean(ekf_rmse)
    eskf_mean_rmse = np.mean(eskf_rmse)
    ukf_mean_rmse = np.mean(ukf_rmse)
    trukf_mean_rmse = np.mean(trukf_rmse)
    
    # 计算平均计算时延
    ekf_mean_time = np.mean(ekf_time) if ekf_time else 0
    eskf_mean_time = np.mean(eskf_time) if eskf_time else 0
    ukf_mean_time = np.mean(ukf_time) if ukf_time else 0
    trukf_mean_time = np.mean(trukf_time) if trukf_time else 0
    
    # 计算平均NEES和NIS
    ekf_mean_nees = np.mean([np.mean(nees) for nees in ekf_nees]) if ekf_nees else 0
    eskf_mean_nees = np.mean([np.mean(nees) for nees in eskf_nees]) if eskf_nees else 0
    ukf_mean_nees = np.mean([np.mean(nees) for nees in ukf_nees]) if ukf_nees else 0
    trukf_mean_nees = np.mean([np.mean(nees) for nees in trukf_nees]) if trukf_nees else 0
    
    ekf_mean_nis = np.mean([np.mean(nis) for nis in ekf_nis]) if ekf_nis else 0
    eskf_mean_nis = np.mean([np.mean(nis) for nis in eskf_nis]) if eskf_nis else 0
    ukf_mean_nis = np.mean([np.mean(nis) for nis in ukf_nis]) if ukf_nis else 0
    trukf_mean_nis = np.mean([np.mean(nis) for nis in trukf_nis]) if trukf_nis else 0
    
    # 计算平均噪声敏感系数
    ekf_mean_noise_sensitivity = np.mean(ekf_noise_sensitivity)
    eskf_mean_noise_sensitivity = np.mean(eskf_noise_sensitivity)
    ukf_mean_noise_sensitivity = np.mean(ukf_noise_sensitivity)
    trukf_mean_noise_sensitivity = np.mean(trukf_noise_sensitivity)
    
    # 计算自相关函数95%置信区间
    # 对于白噪声，95%置信区间约为±1.96/sqrt(N)
    n_samples = len(t) - 1
    confidence_interval = 1.96 / np.sqrt(n_samples)
    
    # 检查自相关函数是否在置信区间内（一致性检验）
    ekf_autocorr_consistency = []
    eskf_autocorr_consistency = []
    ukf_autocorr_consistency = []
    trukf_autocorr_consistency = []
    
    for sim_idx in range(len(ekf_autocorr)):
        for dim in range(3):
            # 跳过lag=0（自相关为1）
            ekf_in_bounds = np.mean(np.abs(ekf_autocorr[sim_idx][dim][1:]) < confidence_interval)
            eskf_in_bounds = np.mean(np.abs(eskf_autocorr[sim_idx][dim][1:]) < confidence_interval)
            ukf_in_bounds = np.mean(np.abs(ukf_autocorr[sim_idx][dim][1:]) < confidence_interval)
            trukf_in_bounds = np.mean(np.abs(trukf_autocorr[sim_idx][dim][1:]) < confidence_interval)
            
            ekf_autocorr_consistency.append(ekf_in_bounds)
            eskf_autocorr_consistency.append(eskf_in_bounds)
            ukf_autocorr_consistency.append(ukf_in_bounds)
            trukf_autocorr_consistency.append(trukf_in_bounds)
    
    ekf_mean_autocorr_consistency = np.mean(ekf_autocorr_consistency) if ekf_autocorr_consistency else 0
    eskf_mean_autocorr_consistency = np.mean(eskf_autocorr_consistency) if eskf_autocorr_consistency else 0
    ukf_mean_autocorr_consistency = np.mean(ukf_autocorr_consistency) if ukf_autocorr_consistency else 0
    trukf_mean_autocorr_consistency = np.mean(trukf_autocorr_consistency) if trukf_autocorr_consistency else 0
    
    # 创建汇总DataFrame
    summary_data = {
        'Algorithm': ['EKF', 'ESKF', 'UKF', 'TRUKF'],
        'Mean_Error_X': [ekf_mean_error[0], eskf_mean_error[0], ukf_mean_error[0], trukf_mean_error[0]],
        'Mean_Error_Y': [ekf_mean_error[1], eskf_mean_error[1], ukf_mean_error[1], trukf_mean_error[1]],
        'Mean_Error_Z': [ekf_mean_error[2], eskf_mean_error[2], ukf_mean_error[2], trukf_mean_error[2]],
        'Std_Error_X': [ekf_std_error[0], eskf_std_error[0], ukf_std_error[0], trukf_std_error[0]],
        'Std_Error_Y': [ekf_std_error[1], eskf_std_error[1], ukf_std_error[1], trukf_std_error[1]],
        'Std_Error_Z': [ekf_std_error[2], eskf_std_error[2], ukf_std_error[2], trukf_std_error[2]],
        'Mean_RMSE': [ekf_mean_rmse, eskf_mean_rmse, ukf_mean_rmse, trukf_mean_rmse],
        'Mean_NEES': [ekf_mean_nees, eskf_mean_nees, ukf_mean_nees, trukf_mean_nees],
        'Mean_NIS': [ekf_mean_nis, eskf_mean_nis, ukf_mean_nis, trukf_mean_nis],
        'Mean_Computation_Time': [ekf_mean_time, eskf_mean_time, ukf_mean_time, trukf_mean_time],
        'Autocorr_Consistency': [ekf_mean_autocorr_consistency, eskf_mean_autocorr_consistency, ukf_mean_autocorr_consistency, trukf_mean_autocorr_consistency],
        'Noise_Sensitivity': [ekf_mean_noise_sensitivity, eskf_mean_noise_sensitivity, ukf_mean_noise_sensitivity, trukf_mean_noise_sensitivity]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # 将所有仿真结果保存到Excel
    with pd.ExcelWriter(os.path.join(results_dir, 'monte_carlo_results_1000_0.3.xlsx')) as writer:
        # 汇总表
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # NEES和NIS统计数据
        nees_data = {
            'Statistic': ['Mean', 'Std', 'Min', 'Max', '95% CI Lower', '95% CI Upper'],
            'EKF_NEES': [np.mean([np.mean(nees) for nees in ekf_nees]) if ekf_nees else 0,
                        np.std([np.mean(nees) for nees in ekf_nees]) if ekf_nees else 0,
                        np.min([np.mean(nees) for nees in ekf_nees]) if ekf_nees else 0,
                        np.max([np.mean(nees) for nees in ekf_nees]) if ekf_nees else 0,
                        np.percentile([np.mean(nees) for nees in ekf_nees], 2.5) if ekf_nees else 0,
                        np.percentile([np.mean(nees) for nees in ekf_nees], 97.5) if ekf_nees else 0],
            'ESKF_NEES': [np.mean([np.mean(nees) for nees in eskf_nees]) if eskf_nees else 0,
                         np.std([np.mean(nees) for nees in eskf_nees]) if eskf_nees else 0,
                         np.min([np.mean(nees) for nees in eskf_nees]) if eskf_nees else 0,
                         np.max([np.mean(nees) for nees in eskf_nees]) if eskf_nees else 0,
                         np.percentile([np.mean(nees) for nees in eskf_nees], 2.5) if eskf_nees else 0,
                         np.percentile([np.mean(nees) for nees in eskf_nees], 97.5) if eskf_nees else 0],
            'UKF_NEES': [np.mean([np.mean(nees) for nees in ukf_nees]) if ukf_nees else 0,
                        np.std([np.mean(nees) for nees in ukf_nees]) if ukf_nees else 0,
                        np.min([np.mean(nees) for nees in ukf_nees]) if ukf_nees else 0,
                        np.max([np.mean(nees) for nees in ukf_nees]) if ukf_nees else 0,
                        np.percentile([np.mean(nees) for nees in ukf_nees], 2.5) if ukf_nees else 0,
                        np.percentile([np.mean(nees) for nees in ukf_nees], 97.5) if ukf_nees else 0],
            'TRUKF_NEES': [np.mean([np.mean(nees) for nees in trukf_nees]) if trukf_nees else 0,
                          np.std([np.mean(nees) for nees in trukf_nees]) if trukf_nees else 0,
                          np.min([np.mean(nees) for nees in trukf_nees]) if trukf_nees else 0,
                          np.max([np.mean(nees) for nees in trukf_nees]) if trukf_nees else 0,
                          np.percentile([np.mean(nees) for nees in trukf_nees], 2.5) if trukf_nees else 0,
                          np.percentile([np.mean(nees) for nees in trukf_nees], 97.5) if trukf_nees else 0],
            'EKF_NIS': [np.mean([np.mean(nis) for nis in ekf_nis]) if ekf_nis else 0,
                       np.std([np.mean(nis) for nis in ekf_nis]) if ekf_nis else 0,
                       np.min([np.mean(nis) for nis in ekf_nis]) if ekf_nis else 0,
                       np.max([np.mean(nis) for nis in ekf_nis]) if ekf_nis else 0,
                       np.percentile([np.mean(nis) for nis in ekf_nis], 2.5) if ekf_nis else 0,
                       np.percentile([np.mean(nis) for nis in ekf_nis], 97.5) if ekf_nis else 0],
            'ESKF_NIS': [np.mean([np.mean(nis) for nis in eskf_nis]) if eskf_nis else 0,
                        np.std([np.mean(nis) for nis in eskf_nis]) if eskf_nis else 0,
                        np.min([np.mean(nis) for nis in eskf_nis]) if eskf_nis else 0,
                        np.max([np.mean(nis) for nis in eskf_nis]) if eskf_nis else 0,
                        np.percentile([np.mean(nis) for nis in eskf_nis], 2.5) if eskf_nis else 0,
                        np.percentile([np.mean(nis) for nis in eskf_nis], 97.5) if eskf_nis else 0],
            'UKF_NIS': [np.mean([np.mean(nis) for nis in ukf_nis]) if ukf_nis else 0,
                       np.std([np.mean(nis) for nis in ukf_nis]) if ukf_nis else 0,
                       np.min([np.mean(nis) for nis in ukf_nis]) if ukf_nis else 0,
                       np.max([np.mean(nis) for nis in ukf_nis]) if ukf_nis else 0,
                       np.percentile([np.mean(nis) for nis in ukf_nis], 2.5) if ukf_nis else 0,
                       np.percentile([np.mean(nis) for nis in ukf_nis], 97.5) if ukf_nis else 0],
            'TRUKF_NIS': [np.mean([np.mean(nis) for nis in trukf_nis]) if trukf_nis else 0,
                         np.std([np.mean(nis) for nis in trukf_nis]) if trukf_nis else 0,
                         np.min([np.mean(nis) for nis in trukf_nis]) if trukf_nis else 0,
                         np.max([np.mean(nis) for nis in trukf_nis]) if trukf_nis else 0,
                         np.percentile([np.mean(nis) for nis in trukf_nis], 2.5) if trukf_nis else 0,
                         np.percentile([np.mean(nis) for nis in trukf_nis], 97.5) if trukf_nis else 0]
        }
        df_nees_nis = pd.DataFrame(nees_data)
        df_nees_nis.to_excel(writer, sheet_name='NEES_NIS_Statistics', index=False)
        
        # 计算时延和噪声敏感系数
        time_noise_data = {
            'Algorithm': ['EKF', 'ESKF', 'UKF', 'TRUKF'],
            'Mean_Computation_Time': [ekf_mean_time, eskf_mean_time, ukf_mean_time, trukf_mean_time],
            'Std_Computation_Time': [np.std(ekf_time) if ekf_time else 0, np.std(eskf_time) if eskf_time else 0, 
                                   np.std(ukf_time) if ukf_time else 0, np.std(trukf_time) if trukf_time else 0],
            'Mean_Noise_Sensitivity': [ekf_mean_noise_sensitivity, eskf_mean_noise_sensitivity, 
                                     ukf_mean_noise_sensitivity, trukf_mean_noise_sensitivity],
            'Std_Noise_Sensitivity': [np.std(ekf_noise_sensitivity), np.std(eskf_noise_sensitivity), 
                                    np.std(ukf_noise_sensitivity), np.std(trukf_noise_sensitivity)],
            'Autocorr_Consistency': [ekf_mean_autocorr_consistency, eskf_mean_autocorr_consistency, 
                                   ukf_mean_autocorr_consistency, trukf_mean_autocorr_consistency]
        }
        df_time_noise = pd.DataFrame(time_noise_data)
        df_time_noise.to_excel(writer, sheet_name='Time_Noise_Statistics', index=False)
        
        # 单次仿真结果（仅保存前100次仿真的详细结果，避免Excel文件过大）
        max_sims_to_save = min(100, num_simulations)
        for i in range(max_sims_to_save):
            sim_data = {
                'Algorithm': ['EKF', 'ESKF', 'UKF', 'TRUKF'],
                'Error_X': [ekf_errors[i, 0], eskf_errors[i, 0], ukf_errors[i, 0], trukf_errors[i, 0]],
                'Error_Y': [ekf_errors[i, 1], eskf_errors[i, 1], ukf_errors[i, 1], trukf_errors[i, 1]],
                'Error_Z': [ekf_errors[i, 2], eskf_errors[i, 2], ukf_errors[i, 2], trukf_errors[i, 2]],
                'RMSE': [ekf_rmse[i], eskf_rmse[i], ukf_rmse[i], trukf_rmse[i]],
                'Noise_Sensitivity': [ekf_noise_sensitivity[i], eskf_noise_sensitivity[i], 
                                    ukf_noise_sensitivity[i], trukf_noise_sensitivity[i]]
            }
            df_sim = pd.DataFrame(sim_data)
            df_sim.to_excel(writer, sheet_name=f'Simulation_{i+1}', index=False)
    
    print(f"蒙特卡洛仿真完成。结果已保存到 {os.path.join(results_dir, 'monte_carlo_results_1000_0.3.xlsx')}")
    
    # 保存第一次仿真的轨迹数据到单独的Excel文件
    trajectory_file = os.path.join(results_dir, 'monte_carlo_trajectories_1000_0.3.xlsx')
    print(f"轨迹数据已保存到 {trajectory_file}")
    
    return {
        'ekf_errors': ekf_errors,
        'eskf_errors': eskf_errors,
        'ukf_errors': ukf_errors,
        'trukf_errors': trukf_errors,
        'ekf_rmse': ekf_rmse,
        'eskf_rmse': eskf_rmse,
        'ukf_rmse': ukf_rmse,
        'trukf_rmse': trukf_rmse,
        'ekf_nees': ekf_nees,
        'eskf_nees': eskf_nees,
        'ukf_nees': ukf_nees,
        'trukf_nees': trukf_nees,
        'ekf_nis': ekf_nis,
        'eskf_nis': eskf_nis,
        'ukf_nis': ukf_nis,
        'trukf_nis': trukf_nis,
        'ekf_time': ekf_time,
        'eskf_time': eskf_time,
        'ukf_time': ukf_time,
        'trukf_time': trukf_time,
        'ekf_noise_sensitivity': ekf_noise_sensitivity,
        'eskf_noise_sensitivity': eskf_noise_sensitivity,
        'ukf_noise_sensitivity': ukf_noise_sensitivity,
        'trukf_noise_sensitivity': trukf_noise_sensitivity,
        'ekf_autocorr': ekf_autocorr,
        'eskf_autocorr': eskf_autocorr,
        'ukf_autocorr': ukf_autocorr,
        'trukf_autocorr': trukf_autocorr
    }

def plot_monte_carlo_results(results):
    """
    绘制蒙特卡洛仿真结果
    
    参数:
        results: 包含仿真结果的字典
    """
    # 提取结果
    ekf_rmse = results['ekf_rmse']
    eskf_rmse = results['eskf_rmse']
    ukf_rmse = results['ukf_rmse']
    trukf_rmse = results['trukf_rmse']
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. 绘制RMSE箱线图和条形图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.boxplot([ekf_rmse, eskf_rmse, ukf_rmse, trukf_rmse], labels=['EKF', 'ESKF', 'UKF', 'TRUKF'])
    plt.title('蒙特卡洛仿真中的RMSE分布')
    plt.ylabel('RMSE (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(222)
    algorithms = ['EKF', 'ESKF', 'UKF', 'TRUKF']
    mean_rmse = [np.mean(ekf_rmse), np.mean(eskf_rmse), np.mean(ukf_rmse), np.mean(trukf_rmse)]
    std_rmse = [np.std(ekf_rmse), np.std(eskf_rmse), np.std(ukf_rmse), np.std(trukf_rmse)]
    
    plt.bar(algorithms, mean_rmse, yerr=std_rmse, capsize=10, alpha=0.7)
    plt.title('平均RMSE及其标准差')
    plt.ylabel('RMSE (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 绘制NEES箱线图
    plt.subplot(223)
    ekf_nees_means = [np.mean(nees) for nees in results['ekf_nees']] if results['ekf_nees'] else []
    eskf_nees_means = [np.mean(nees) for nees in results['eskf_nees']] if results['eskf_nees'] else []
    ukf_nees_means = [np.mean(nees) for nees in results['ukf_nees']] if results['ukf_nees'] else []
    trukf_nees_means = [np.mean(nees) for nees in results['trukf_nees']] if results['trukf_nees'] else []
    
    nees_data = [ekf_nees_means, eskf_nees_means, ukf_nees_means, trukf_nees_means]
    nees_data = [x for x in nees_data if x]  # 过滤空列表
    
    if nees_data:
        plt.boxplot(nees_data, labels=['EKF', 'ESKF', 'UKF', 'TRUKF'][:len(nees_data)])
        plt.title('归一化估计误差平方(NEES)分布')
        plt.ylabel('NEES')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加理想值参考线（对于3维位置，理想NEES为3）
        plt.axhline(y=3, color='r', linestyle='--', label='理想值(3)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'NEES数据不可用', horizontalalignment='center', verticalalignment='center')
    
    # 3. 绘制计算时延条形图
    plt.subplot(224)
    mean_time = [np.mean(results['ekf_time']) if results['ekf_time'] else 0,
                np.mean(results['eskf_time']) if results['eskf_time'] else 0,
                np.mean(results['ukf_time']) if results['ukf_time'] else 0,
                np.mean(results['trukf_time']) if results['trukf_time'] else 0]
    
    std_time = [np.std(results['ekf_time']) if results['ekf_time'] else 0,
               np.std(results['eskf_time']) if results['eskf_time'] else 0,
               np.std(results['ukf_time']) if results['ukf_time'] else 0,
               np.std(results['trukf_time']) if results['trukf_time'] else 0]
    
    plt.bar(algorithms, mean_time, yerr=std_time, capsize=10, alpha=0.7)
    plt.title('平均计算时延')
    plt.ylabel('时间 (秒)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'monte_carlo_rmse_nees_comparison_1000_0.3.png'), dpi=300)
    
    # 4. 绘制噪声敏感系数和自相关一致性
    plt.figure(figsize=(15, 6))
    
    plt.subplot(121)
    noise_sensitivity = [results['ekf_noise_sensitivity'], results['eskf_noise_sensitivity'],
                        results['ukf_noise_sensitivity'], results['trukf_noise_sensitivity']]
    plt.boxplot(noise_sensitivity, labels=algorithms)
    plt.title('噪声敏感系数分布')
    plt.ylabel('噪声敏感系数')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(122)
    # 提取自相关函数一致性数据
    ekf_autocorr = results['ekf_autocorr']
    eskf_autocorr = results['eskf_autocorr']
    ukf_autocorr = results['ukf_autocorr']
    trukf_autocorr = results['trukf_autocorr']
    
    # 计算自相关函数95%置信区间
    if ekf_autocorr and len(ekf_autocorr) > 0 and len(ekf_autocorr[0]) > 0 and len(ekf_autocorr[0][0]) > 0:
        # 使用第一次仿真的第一个维度的自相关函数长度
        n_samples = len(ekf_autocorr[0][0])
        confidence_interval = 1.96 / np.sqrt(n_samples)
        
        # 计算每个算法的自相关函数平均值（跨所有仿真和维度）
        max_lag = min(20, len(ekf_autocorr[0][0]) if ekf_autocorr and ekf_autocorr[0] else 0)
        lags = np.arange(max_lag)
        
        # 计算平均自相关函数
        ekf_mean_acorr = np.zeros(max_lag)
        eskf_mean_acorr = np.zeros(max_lag)
        ukf_mean_acorr = np.zeros(max_lag)
        trukf_mean_acorr = np.zeros(max_lag)
        
        count = 0
        for sim_idx in range(min(100, len(ekf_autocorr))):
            for dim in range(3):
                if sim_idx < len(ekf_autocorr) and dim < len(ekf_autocorr[sim_idx]) and len(ekf_autocorr[sim_idx][dim]) >= max_lag:
                    ekf_mean_acorr += ekf_autocorr[sim_idx][dim][:max_lag]
                    eskf_mean_acorr += eskf_autocorr[sim_idx][dim][:max_lag]
                    ukf_mean_acorr += ukf_autocorr[sim_idx][dim][:max_lag]
                    trukf_mean_acorr += trukf_autocorr[sim_idx][dim][:max_lag]
                    count += 1
        
        if count > 0:
            ekf_mean_acorr /= count
            eskf_mean_acorr /= count
            ukf_mean_acorr /= count
            trukf_mean_acorr /= count
            
            plt.plot(lags, ekf_mean_acorr, 'o-', label='EKF')
            plt.plot(lags, eskf_mean_acorr, 's-', label='ESKF')
            plt.plot(lags, ukf_mean_acorr, '^-', label='UKF')
            plt.plot(lags, trukf_mean_acorr, 'd-', label='TRUKF')
            
            # 添加95%置信区间
            plt.axhspan(-confidence_interval, confidence_interval, alpha=0.2, color='gray', label='95% 置信区间')
            
            plt.title('平均自相关函数')
            plt.xlabel('滞后')
            plt.ylabel('自相关')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        else:
            plt.text(0.5, 0.5, '自相关数据不可用', horizontalalignment='center', verticalalignment='center')
    else:
        plt.text(0.5, 0.5, '自相关数据不可用', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'monte_carlo_noise_autocorr_1000_0.3.png'), dpi=300)
    
    # 5. 绘制NIS分布
    plt.figure(figsize=(10, 6))
    ekf_nis_means = [np.mean(nis) for nis in results['ekf_nis']] if results['ekf_nis'] else []
    eskf_nis_means = [np.mean(nis) for nis in results['eskf_nis']] if results['eskf_nis'] else []
    ukf_nis_means = [np.mean(nis) for nis in results['ukf_nis']] if results['ukf_nis'] else []
    trukf_nis_means = [np.mean(nis) for nis in results['trukf_nis']] if results['trukf_nis'] else []
    
    nis_data = [ekf_nis_means, eskf_nis_means, ukf_nis_means, trukf_nis_means]
    nis_data = [x for x in nis_data if x]  # 过滤空列表
    
    if nis_data:
        plt.boxplot(nis_data, labels=['EKF', 'ESKF', 'UKF', 'TRUKF'][:len(nis_data)])
        plt.title('归一化新息平方(NIS)分布')
        plt.ylabel('NIS')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 对于不同的测量维度，理想NIS值不同，这里假设平均维度为3
        plt.axhline(y=3, color='r', linestyle='--', label='理想值(3)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'NIS数据不可用', horizontalalignment='center', verticalalignment='center')

def main():
    # 运行蒙特卡洛仿真
    results = run_monte_carlo_simulation(num_simulations=1000)
    
    # 绘制结果
    plot_monte_carlo_results(results)
    
    print("所有图表已保存到results目录")
    plt.show()

if __name__ == "__main__":
    main()