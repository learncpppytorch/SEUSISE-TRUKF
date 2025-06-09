import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# 导入自定义模块
from params import EnhancedSensorParams
from trajectory.trajectory import generate_true_trajectory
from error_models import EnhancedErrorModels
from filters.ekf import EKF
from filters.eskf import ESKF
from filters.ukf import UKF
from filters.trukf import TRUKF
def run_monte_carlo_simulation(num_simulations=50):
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
        
        # 主仿真循环
        for i in range(1, n):
            # 获取IMU测量值
            gyro_meas = true_gyro[:, i] + gyro_err[i]
            accel_meas = true_accel[:, i] + accel_err[i]
            
            # EKF预测和更新
            ekf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz USBL更新
                ekf.update(usbl_meas[:, i], 'USBL')
            if i % int(0.1/params.dt) == 0:  # 10Hz DVL和深度更新
                ekf.update(dvl_meas[:, i], 'DVL')
                ekf.update(np.array([depth_meas[i]]), 'Depth')
            ekf_pos[:, i] = ekf.pos.copy()
            
            # ESKF预测和更新
            eskf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz USBL更新
                eskf.update(usbl_meas[:, i], 'USBL')
            if i % int(0.1/params.dt) == 0:  # 10Hz DVL和深度更新
                eskf.update(dvl_meas[:, i], 'DVL')
                eskf.update(np.array([depth_meas[i]]), 'Depth')
            eskf_pos[:, i] = eskf.pos_nominal.copy()
            
            # UKF预测和更新
            ukf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz更新
                if not np.isnan(depth_meas[i]):
                    ukf.update(np.array([depth_meas[i]]), 'Depth', i)
                if not np.any(np.isnan(usbl_meas[:, i])):
                    ukf.update(usbl_meas[:, i], 'USBL', i)
                if not np.any(np.isnan(dvl_meas[:, i])):
                    ukf.update(dvl_meas[:, i], 'DVL', i)
            ukf_pos[:, i] = ukf.pos_nominal.copy()
            
            # TRUKF预测和更新
            trukf.predict(params.dt, gyro_meas, accel_meas)
            if i % int(1.0/params.dt) == 0:  # 1Hz更新
                if not np.isnan(depth_meas[i]):
                    trukf.update(np.array([depth_meas[i]]), 'Depth', i)
                if not np.any(np.isnan(usbl_meas[:, i])):
                    trukf.update(usbl_meas[:, i], 'USBL', i)
                if not np.any(np.isnan(dvl_meas[:, i])):
                    trukf.update(dvl_meas[:, i], 'DVL', i)
            trukf_pos[:, i] = trukf.get_position().copy()
        
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
    ukf_mean_error = np.mean(np.abs(eskf_errors), axis=0)
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
    
    # 创建汇总DataFrame
    summary_data = {
        'Algorithm': ['EKF', 'ESKF', 'UKF', 'TRUKF'],
        'Mean_Error_X': [ekf_mean_error[0], eskf_mean_error[0], ukf_mean_error[0], trukf_mean_error[0]],
        'Mean_Error_Y': [ekf_mean_error[1], eskf_mean_error[1], ukf_mean_error[1], trukf_mean_error[1]],
        'Mean_Error_Z': [ekf_mean_error[2], eskf_mean_error[2], ukf_mean_error[2], trukf_mean_error[2]],
        'Std_Error_X': [ekf_std_error[0], eskf_std_error[0], ukf_std_error[0], trukf_std_error[0]],
        'Std_Error_Y': [ekf_std_error[1], eskf_std_error[1], ukf_std_error[1], trukf_std_error[1]],
        'Std_Error_Z': [ekf_std_error[2], eskf_std_error[2], ukf_std_error[2], trukf_std_error[2]],
        'Mean_RMSE': [ekf_mean_rmse, eskf_mean_rmse, ukf_mean_rmse, trukf_mean_rmse]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # 将所有仿真结果保存到Excel
    with pd.ExcelWriter(os.path.join(results_dir, 'monte_carlo_results_0.5.xlsx')) as writer:
        # 汇总表
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # 单次仿真结果
        for i in range(num_simulations):
            sim_data = {
                'Algorithm': ['EKF', 'ESKF', 'UKF', 'TRUKF'],
                'Error_X': [ekf_errors[i, 0], eskf_errors[i, 0], ukf_errors[i, 0], trukf_errors[i, 0]],
                'Error_Y': [ekf_errors[i, 1], eskf_errors[i, 1], ukf_errors[i, 1], trukf_errors[i, 1]],
                'Error_Z': [ekf_errors[i, 2], eskf_errors[i, 2], ukf_errors[i, 2], trukf_errors[i, 2]],
                'RMSE': [ekf_rmse[i], eskf_rmse[i], ukf_rmse[i], trukf_rmse[i]]
            }
            df_sim = pd.DataFrame(sim_data)
            df_sim.to_excel(writer, sheet_name=f'Simulation_{i+1}', index=False)
    
    print(f"蒙特卡洛仿真完成。结果已保存到 {os.path.join(results_dir, 'monte_carlo_results_0.5.xlsx')}")
    
    return {
        'ekf_errors': ekf_errors,
        'eskf_errors': eskf_errors,
        'ukf_errors': ukf_errors,
        'trukf_errors': trukf_errors,
        'ekf_rmse': ekf_rmse,
        'eskf_rmse': eskf_rmse,
        'ukf_rmse': ukf_rmse,
        'trukf_rmse': trukf_rmse
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
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制RMSE箱线图
    plt.subplot(121)
    plt.boxplot([ekf_rmse, eskf_rmse, ukf_rmse, trukf_rmse], labels=['EKF', 'ESKF', 'UKF', 'TRUKF'])
    plt.title('蒙特卡洛仿真中的RMSE分布')
    plt.ylabel('RMSE (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制平均RMSE条形图
    plt.subplot(122)
    algorithms = ['EKF', 'ESKF', 'UKF', 'TRUKF']
    mean_rmse = [np.mean(ekf_rmse), np.mean(eskf_rmse), np.mean(ukf_rmse), np.mean(trukf_rmse)]
    std_rmse = [np.std(ekf_rmse), np.std(eskf_rmse), np.std(ukf_rmse), np.std(trukf_rmse)]
    
    plt.bar(algorithms, mean_rmse, yerr=std_rmse, capsize=10, alpha=0.7)
    plt.title('平均RMSE及其标准差')
    plt.ylabel('RMSE (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'monte_carlo_rmse_comparison_0.5.png'), dpi=300)
    plt.show()

def main():
    # 运行蒙特卡洛仿真
    results = run_monte_carlo_simulation(num_simulations=50)
    
    # 绘制结果
    plot_monte_carlo_results(results)

if __name__ == "__main__":
    main()