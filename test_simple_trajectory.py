##用于测试生成轨迹的代码##
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from trajectory.trajectory import generate_true_trajectory
from params import EnhancedSensorParams
import os
# 创建保存结果的文件夹
output_dir = "trajectory_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化参数
params = EnhancedSensorParams()

# 生成轨迹
true_pos, true_vel, true_gyro, true_accel, t = generate_true_trajectory(params)

# 创建3D图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(true_pos[0], true_pos[1], true_pos[2], 'b-', linewidth=2)

# 标记起点和终点
ax.scatter(true_pos[0, 0], true_pos[1, 0], true_pos[2, 0], c='g', marker='o', s=100, label='起点')
ax.scatter(true_pos[0, -1], true_pos[1, -1], true_pos[2, -1], c='r', marker='o', s=100, label='终点')

# 设置坐标轴标签
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('简化水下导航轨迹')
ax.legend()

# 保存3D轨迹图
plt.savefig(os.path.join(output_dir, 'simple_trajectory_3d.png'))
plt.close()

# 创建速度图
plt.figure(figsize=(12, 8))

# 绘制速度
plt.subplot(3, 1, 1)
plt.plot(t, true_vel[0], 'r-')
plt.ylabel('Vx (m/s)')
plt.title('速度分量')

plt.subplot(3, 1, 2)
plt.plot(t, true_vel[1], 'g-')
plt.ylabel('Vy (m/s)')

plt.subplot(3, 1, 3)
plt.plot(t, true_vel[2], 'b-')
plt.ylabel('Vz (m/s)')
plt.xlabel('时间 (s)')

# 保存速度图
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'simple_trajectory_velocity.png'))
plt.close()

# 添加加速度分析
plt.figure(figsize=(12, 8))
accel = np.diff(true_vel, axis=1) / params.dt
plt.subplot(3, 1, 1)
plt.plot(t[:-1], accel[0], 'r-')
plt.ylabel('ax (m/s²)')
plt.title('加速度分量')

plt.subplot(3, 1, 2)
plt.plot(t[:-1], accel[1], 'g-')
plt.ylabel('ay (m/s²)')

plt.subplot(3, 1, 3)
plt.plot(t[:-1], accel[2], 'b-')
plt.ylabel('az (m/s²)')
plt.xlabel('时间 (s)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'acceleration_analysis.png'))
plt.close()

# 添加轨迹曲率分析
def calculate_curvature(pos):
    # 计算轨迹曲率
    dx = np.diff(pos[0])
    dy = np.diff(pos[1])
    dz = np.diff(pos[2])
    
    d2x = np.diff(dx)
    d2y = np.diff(dy)
    d2z = np.diff(dz)
    
    dr = np.sqrt(dx[:-1]**2 + dy[:-1]**2 + dz[:-1]**2)
    d2r = np.sqrt(d2x**2 + d2y**2 + d2z**2)
    
    curvature = d2r / (dr**2)
    return curvature

curvature = calculate_curvature(true_pos)
plt.figure(figsize=(10, 6))
plt.plot(t[2:], curvature)
plt.title('轨迹曲率分析')
plt.xlabel('时间 (s)')
plt.ylabel('曲率 (1/m)')
plt.savefig(os.path.join(output_dir, 'curvature_analysis.png'))
plt.close()

# 检查轨迹连续性
def check_continuity(pos, vel, dt):
    # 计算位置差分
    pos_diff = np.diff(pos, axis=1)
    # 计算速度积分（理论位移）
    vel_int = vel[:, :-1] * dt
    # 计算误差
    error = np.abs(pos_diff - vel_int)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    # 将结果写入文件
    with open(os.path.join(output_dir, 'continuity_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write('位置连续性检查：\n')
        f.write(f'  最大误差: {max_error:.6f} m\n')
        f.write(f'  平均误差: {mean_error:.6f} m\n')
        
        # 检查速度连续性（突变）
        vel_diff = np.diff(vel, axis=1) / dt
        max_accel = np.max(np.abs(vel_diff))
        f.write('\n速度连续性检查：\n')
        f.write(f'  最大加速度: {max_accel:.6f} m/s²\n')
    
    return max_error, mean_error, max_accel

print('\n轨迹连续性检查：')
max_error, mean_error, max_accel = check_continuity(true_pos, true_vel, params.dt)
print(f'分析结果已保存至 {output_dir} 文件夹')
