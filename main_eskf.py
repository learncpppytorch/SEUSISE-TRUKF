import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

from params import EnhancedSensorParams
from trajectory.trajectory import generate_true_trajectory
from error_models import EnhancedErrorModels
from filters.eskf import ESKF

def main():
   
    params = EnhancedSensorParams()
    
    # Select data mode: 'simulation' or 'real'
    data_mode = 'simulation'
    
    if data_mode == 'simulation':
        # ===== Simulation Data Mode =====
        # Generate simulation trajectory
        true_pos, true_vel, true_gyro, true_accel, t = generate_true_trajectory(params)
        n = len(t)
        
        print(f"Trajectory start point: {true_pos[:,0]}")
        print(f"Trajectory end point: {true_pos[:,-1]}")
        
        gyro_err, accel_err = EnhancedErrorModels.sins_error_model(
            np.zeros((n,3)), true_vel, params)
        
        depth_meas = EnhancedErrorModels.depth_measurement(
            true_pos[2], 10+0.1*np.random.randn(n), params)
        dvl_meas = EnhancedErrorModels.dvl_measurement(true_vel, true_pos[2], t, params)
        usbl_meas = EnhancedErrorModels.usbl_measurement(true_pos, t, params)
    
    else:
        # ===== Real Data Mode =====
        print("Loading real data...")
        # Load reference data (true trajectory)
        data_ref = sio.loadmat('realworldData/sensorData_ref_9.mat')
        # Load sensor data
        data = sio.loadmat('realworldData/sensorData_9.mat')
        
        true_pos = np.zeros((3, data_ref['timestamps_ref'].shape[0]))
        
        lat_ref = data_ref['lat_ref'].flatten()
        lon_ref = data_ref['lon_ref'].flatten()
     
        earth_radius = 6371000  # Earth radius in meters
        lat0, lon0 = lat_ref[0], lon_ref[0]  # Starting point as origin
        true_pos[0, :] = (lon_ref - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
        true_pos[1, :] = (lat_ref - lat0) * np.pi * earth_radius / 180.0
        # Check att_ref structure and extract Z coordinate
        print(f"att_ref shape: {data_ref['att_ref'].shape}")
        if 'GNSSReading_diff' in data['sensorData'][0][0].dtype.names:
            
            initial_depth = data['sensorData'][0][0]['GNSSReading_diff'][0, 0][0, 2] if data['sensorData'][0][0]['GNSSReading_diff'][0, 0].size > 0 else 0
            true_pos[2, :] = -initial_depth  
        else:
            true_pos[2, :] = 0  
        # Extract true velocity
        true_vel = data_ref['Velocity_ref'].T  
        
        # Extract timestamps
        t = data_ref['timestamps_ref'].flatten()
        n = len(t)
        
        # Extract sensor data
        sensor_data = data['sensorData'][0]
        
        # Initial state (first row of data)
        initial_state = sensor_data[0]
        
        gyro_readings = np.zeros((3, n))
        accel_readings = np.zeros((3, n))
        
        usbl_meas = np.zeros((3, n))
        
        dvl_meas = np.zeros((3, n))
        
        depth_meas = np.zeros(n)
        
        # Fill sensor data
        for i in range(1, n):
            if i < len(sensor_data):
                if 'GyroReadings' in sensor_data[i].dtype.names and sensor_data[i]['GyroReadings'][0, 0].size > 0:
                    try:
                        # Get raw data
                        gyro_data = sensor_data[i]['GyroReadings'][0, 0]
                        # Check data shape
                        if gyro_data.size == 3 or len(gyro_data.shape) == 1 and gyro_data.shape[0] == 3:
                            # Normal 3D data
                            gyro_readings[:, i] = gyro_data.flatten()
                        elif len(gyro_data.shape) == 1 and gyro_data.shape[0] > 3:
                          
                            print(f"Note: Gyro data shape abnormal at step {i} ({gyro_data.shape}), taking first 3 elements")
                            gyro_readings[:, i] = gyro_data[:3]
                        elif len(gyro_data.shape) > 1:
                            if len(gyro_data.shape) == 2 and gyro_data.shape[0] >= 1 and gyro_data.shape[1] >= 3:
                                
                                gyro_readings[:, i] = np.mean(gyro_data[:, :3], axis=0)
                            else:
                                # Use previous valid value if cannot process
                                if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                                    gyro_readings[:, i] = gyro_readings[:, i-1]
                        else:
                            print(f"Warning: Cannot process gyro data at step {i}, shape: {gyro_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing gyro data at step {i}: {e}")
                        # Use previous valid value
                        if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                            gyro_readings[:, i] = gyro_readings[:, i-1]
                
                if 'AccelReadings' in sensor_data[i].dtype.names and sensor_data[i]['AccelReadings'][0, 0].size > 0:
                    try:
                        accel_data = sensor_data[i]['AccelReadings'][0, 0]
                   
                        if accel_data.size == 3 or len(accel_data.shape) == 1 and accel_data.shape[0] == 3:
                     
                            accel_readings[:, i] = accel_data.flatten()
                        elif len(accel_data.shape) == 1 and accel_data.shape[0] > 3:
                        
                            print(f"Note: Accelerometer data shape abnormal at step {i} ({accel_data.shape}), taking first 3 elements")
                            accel_readings[:, i] = accel_data[:3]
                        elif len(accel_data.shape) > 1:
                            if len(accel_data.shape) == 2 and accel_data.shape[0] >= 1 and accel_data.shape[1] >= 3:
                            
                                accel_readings[:, i] = np.mean(accel_data[:, :3], axis=0)
                            else:
                                if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                                    accel_readings[:, i] = accel_readings[:, i-1]
                        else:
                            print(f"Warning: Cannot process accelerometer data at step {i}, shape: {accel_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing accelerometer data at step {i}: {e}")
                        
                        if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                            accel_readings[:, i] = accel_readings[:, i-1]
                
                if 'GNSSReading' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading'][0, 0].size > 0:
                    gnss = sensor_data[i]['GNSSReading'][0, 0].flatten()
                    if len(gnss) >= 3:
                        usbl_meas[0, i] = (gnss[1] - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
                        usbl_meas[1, i] = (gnss[0] - lat0) * np.pi * earth_radius / 180.0
                        usbl_meas[2, i] = gnss[2]  # Height
                
                if 'DVL_Velocity' in sensor_data[i].dtype.names and sensor_data[i]['DVL_Velocity'][0, 0].size > 0:
                    try:
                        dvl_data = sensor_data[i]['DVL_Velocity'][0, 0]
                    
                        if dvl_data.size == 3 or len(dvl_data.shape) == 1 and dvl_data.shape[0] == 3:
                            dvl_meas[:, i] = dvl_data.flatten()
                        elif len(dvl_data.shape) == 1 and dvl_data.shape[0] > 3:
                            print(f"Note: DVL data shape abnormal at step {i} ({dvl_data.shape}), taking first 3 elements")
                            dvl_meas[:, i] = dvl_data[:3]
                        else:
                        
                            if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                                dvl_meas[:, i] = dvl_meas[:, i-1]
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing DVL data at step {i}: {e}")
                
                        if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                            dvl_meas[:, i] = dvl_meas[:, i-1]
                
       
                if 'GNSSReading_diff' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading_diff'][0, 0].size > 0:
                    gnss_diff = sensor_data[i]['GNSSReading_diff'][0, 0].flatten()
                    if len(gnss_diff) >= 3:
                        depth_meas[i] = gnss_diff[2]  # Depth value
        
        # Use real IMU data instead of simulation data
        true_gyro = gyro_readings
        true_accel = accel_readings
        
        # Generate IMU errors (these errors are already included in measurements in real data mode)
        gyro_err = np.zeros((n, 3))
        accel_err = np.zeros((n, 3))
    
    # Initialize ESKF
    eskf = ESKF(params)
    est_pos = np.zeros((3, n))
    est_vel = np.zeros((3, n))
    est_att = np.zeros((3, n))
    
    # Set initial state
    eskf.pos_nominal = true_pos[:, 0].copy()  # Use copy() to avoid reference passing
    eskf.vel_nominal = true_vel[:, 0].copy()
    est_pos[:, 0] = eskf.pos_nominal
    est_vel[:, 0] = eskf.vel_nominal
    
    # Main loop
    for i in range(1, n):
        # IMU prediction
        gyro = true_gyro[:, i] + gyro_err[i]
        accel = true_accel[:, i] + accel_err[i]
        eskf.predict(params.dt, gyro, accel)
        
        # Sensor updates
        if i % int(1.0/params.dt) == 0:  
            # USBL update
            eskf.update(usbl_meas[:, i], 'USBL')
            
        if i % int(0.1/params.dt) == 0: 
            # DVL update
            eskf.update(dvl_meas[:, i], 'DVL')
            eskf.update(np.array([depth_meas[i]]), 'Depth')
        
        # Record estimation results
        est_pos[:, i] = eskf.pos_nominal.copy()
        est_vel[:, i] = eskf.vel_nominal.copy()
        est_att[:, i] = eskf.att_nominal.copy()
        
        # Print debug info every 500 steps
        if i % 500 == 0:
            # Calculate position errors in each direction
            pos_error_xyz = true_pos[:, i] - est_pos[:, i]
            pos_error = np.linalg.norm(pos_error_xyz)
            vel_error = np.linalg.norm(true_vel[:, i] - est_vel[:, i])
            
            print(f"\nStep {i}:")
            print(f"Current true position: {true_pos[:,i]}")
            print(f"Current estimated position: {est_pos[:,i]}")
            print(f"Position error: {pos_error:.2f} m")
            print(f"Position error X: {abs(pos_error_xyz[0]):.2f} m")
            print(f"Position error Y: {abs(pos_error_xyz[1]):.2f} m")
            print(f"Position error Z: {abs(pos_error_xyz[2]):.2f} m")
            print(f"Velocity error: {vel_error:.2f} m/s")
    
    # Check final trajectory
    print("\nTrajectory Analysis:")
    print(f"Estimated trajectory start: {est_pos[:,0]}")
    print(f"Estimated trajectory end: {est_pos[:,-1]}")
    print(f"True trajectory start: {true_pos[:,0]}")
    print(f"True trajectory end: {true_pos[:,-1]}")
    
    # *************ESKF Simulation Loop**************
   
    # Visualization and error analysis
    visualize_results(true_pos, est_pos, true_vel, est_vel, dvl_meas, depth_meas, t, 
                      true_gyro, true_accel, usbl_meas, data_mode)
def visualize_results(true_pos, est_pos, true_vel, est_vel, dvl_meas, depth_meas, t, gyro_meas=None, accel_meas=None, usbl_meas=None, data_mode='simulation'):
    """Visualization and error analysis"""
    # Create large figure window
    plt.figure(figsize=(20, 15))
    
    # 1. 3D trajectory comparison (top left)
    ax1 = plt.subplot(221, projection='3d')
    # True trajectory and ESKF estimate
    ax1.plot(true_pos[0], true_pos[1], true_pos[2], 'b', linewidth=2, label='True Trajectory')
    ax1.plot(est_pos[0,30:], est_pos[1,30:], est_pos[2,30:], 'r--', linewidth=2, label='ESKF Estimate')
    
    # Adjust chart title based on data mode
    trajectory_title = '3D Trajectory Comparison' if data_mode == 'simulation' else 'ESKF Navigation Results'
    ax1.set_title(trajectory_title, fontsize=12)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # 2. Position error analysis (top right)
    ax2 = plt.subplot(222)
    # Calculate position errors for X,Y,Z axes
    x_error = np.abs(true_pos[0] - est_pos[0])
    y_error = np.abs(true_pos[1] - est_pos[1])
    z_error = np.abs(true_pos[2] - est_pos[2])
    
    ax2.plot(t, x_error, 'r', label='X-axis Error', linewidth=1.5)
    ax2.plot(t, y_error, 'g', label='Y-axis Error', linewidth=1.5)
    ax2.plot(t, z_error, 'b', label='Z-axis Error', linewidth=1.5)
    ax2.set_title('Position Error Analysis', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Error (m)', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    # 3. DVL velocity measurements (bottom left)
    ax3 = plt.subplot(223)
    # DVL measurements
    valid_dvl = ~np.isnan(dvl_meas[0])
    ax3.plot(t[valid_dvl], dvl_meas[0][valid_dvl], 'r.', label='DVL Vx', markersize=2)
    ax3.plot(t[valid_dvl], dvl_meas[1][valid_dvl], 'g.', label='DVL Vy', markersize=2)
    ax3.plot(t[valid_dvl], dvl_meas[2][valid_dvl], 'b.', label='DVL Vz', markersize=2)
    # True velocity
    ax3.plot(t, true_vel[0], 'r-', label='True Vx', alpha=0.5)
    ax3.plot(t, true_vel[1], 'g-', label='True Vy', alpha=0.5)
    ax3.plot(t, true_vel[2], 'b-', label='True Vz', alpha=0.5)
    ax3.set_title('DVL Measurements vs True Velocity', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Velocity (m/s)', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True)
    
    # 4. Position error analysis (bottom right)
    ax4 = plt.subplot(224)
    # Calculate sensor position errors
    if usbl_meas is not None:
        usbl_error = np.sqrt(np.sum((usbl_meas - true_pos)**2, axis=0))
    depth_error = np.abs(depth_meas - true_pos[2])
    total_pos_error = np.linalg.norm(true_pos - est_pos, axis=0)
    
    # Plot errors
    ax4.plot(t, total_pos_error, 'r', label='Total Position Error', linewidth=1.5)
    
    # Adjust chart content based on data mode
    if data_mode == 'simulation':
        # Show more detailed error analysis in simulation mode
        ax4.plot(t, x_error, 'g', label='X Position Error', linewidth=1.5)
        ax4.plot(t, y_error, 'b', label='Y Position Error', linewidth=1.5)
        ax4.plot(t, z_error, 'y', label='Z Position Error', linewidth=1.5)
    else:
        # Show sensor availability in real data mode
        if usbl_meas is not None:
            valid_usbl = ~np.isnan(usbl_meas[0])
        else:
            valid_usbl = np.zeros_like(t, dtype=bool)
        valid_depth = ~np.isnan(depth_meas)
        valid_dvl = ~np.isnan(dvl_meas[0])
        
        # Create sensor availability indicators
        usbl_avail = np.zeros_like(t)
        depth_avail = np.zeros_like(t)
        dvl_avail = np.zeros_like(t)
        
        usbl_avail[valid_usbl] = 1
        depth_avail[valid_depth] = 0.8
        dvl_avail[valid_dvl] = 0.6
        
        ax4.plot(t, usbl_avail, 'g', label='USBL Available', linewidth=1, alpha=0.5)
        ax4.plot(t, depth_avail, 'b', label='Depth Available', linewidth=1, alpha=0.5)
        ax4.plot(t, dvl_avail, 'y', label='DVL Available', linewidth=1, alpha=0.5)
    
    ax4.set_title('Position Error and Sensor Availability', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Error (m) / Availability', fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical information
    print("\nError Statistical Analysis:")
    print(f"ESKF Position RMS Error: {np.sqrt(np.mean(total_pos_error**2)):.3f} m")
    print(f"Velocity RMS Error: {np.sqrt(np.mean(np.sum((true_vel - est_vel)**2, axis=0))):.3f} m/s")
    
    # Save position data to CSV file
    print("\nSaving position data to CSV file...")
    # Prepare data: timestamps, true positions(XYZ), ESKF estimated positions(XYZ)
    csv_data = np.zeros((len(t), 7))  # 7 columns: time, true X/Y/Z, estimated X/Y/Z
    csv_data[:, 0] = t  # timestamps
    csv_data[:, 1:4] = true_pos.T  # true positions (transposed to match row format)
    csv_data[:, 4:7] = est_pos.T   # ESKF estimated positions (transposed to match row format)
    
    # Set column headers
    header = "Time(s),True Position X(m),True Position Y(m),True Position Z(m),ESKF Estimated Position X(m),ESKF Estimated Position Y(m),ESKF Estimated Position Z(m)"
    
    # Set different filename based on data mode
    filename = "eskf_simulation_results.csv" if data_mode == 'simulation' else "eskf_real_data_results.csv"
    
    # Save to CSV file
    np.savetxt(filename, csv_data, delimiter=",", header=header, comments="")
    print(f"Position data has been saved to '{filename}'")


if __name__ == "__main__":
    main()