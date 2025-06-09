import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

# Import custom modules
from params import EnhancedSensorParams
from trajectory.trajectory import generate_true_trajectory
from error_models import EnhancedErrorModels
from filters.ekf import EKF

def main():
    # Initialize parameters
    params = EnhancedSensorParams()
    
    # Select data mode: 'simulation' or 'real'
    data_mode = 'simulation'
    
    if data_mode == 'simulation':
        # ===== Simulation Mode =====
        # Generate simulated trajectory
        true_pos, true_vel, true_gyro, true_accel, t = generate_true_trajectory(params)
        n = len(t)
        
        # Check trajectory start and end points
        print(f"Trajectory start: {true_pos[:,0]}")
        print(f"Trajectory end: {true_pos[:,-1]}")
        
        # Generate sensor data
        gyro_err, accel_err = EnhancedErrorModels.sins_error_model(
            np.zeros((n,3)), true_vel, params)
        
        # Generate observation data
        depth_meas = true_pos[2] + params.depth_bias + 0.01 * np.random.randn(n)
        dvl_meas = EnhancedErrorModels.dvl_measurement(true_vel, true_pos[2], t, params)
        usbl_meas = EnhancedErrorModels.usbl_measurement(true_pos, t, params)
    
    else:
        # ===== Real Data Mode =====
        print("Loading real data...")
        # Load reference data (true trajectory)
        data_ref = sio.loadmat('realworldData/sensorData_ref_9.mat')
        # Load sensor data
        data = sio.loadmat('realworldData/sensorData_9.mat')
        
        # Extract reference data (true trajectory)
        true_pos = np.zeros((3, data_ref['timestamps_ref'].shape[0]))
        
        lat_ref = data_ref['lat_ref'].flatten()
        lon_ref = data_ref['lon_ref'].flatten()
        earth_radius = 6371000  # Earth radius in meters
        lat0, lon0 = lat_ref[0], lon_ref[0]  # Origin point
        true_pos[0, :] = (lon_ref - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
        true_pos[1, :] = (lat_ref - lat0) * np.pi * earth_radius / 180.0
        
        print(f"att_ref shape: {data_ref['att_ref'].shape}")
       
        if 'GNSSReading_diff' in data['sensorData'][0][0].dtype.names:
           
            initial_depth = data['sensorData'][0][0]['GNSSReading_diff'][0, 0][0, 2] if data['sensorData'][0][0]['GNSSReading_diff'][0, 0].size > 0 else 0
            true_pos[2, :] = -initial_depth  # Use negative depth as Z coordinate
        else:
            true_pos[2, :] = 0  # Use 0 if no depth data available
        
        # Extract true velocity
        true_vel = data_ref['Velocity_ref'].T  # Transpose to match shape (3, n)
        
        # Extract timestamps
        t = data_ref['timestamps_ref'].flatten()
        n = len(t)
        
        # Extract sensor data
        sensor_data = data['sensorData'][0]
        
        # Initial state (first row of data)
        initial_state = sensor_data[0]
        
        # Extract IMU data (gyroscope and accelerometer)
        gyro_readings = np.zeros((3, n))
        accel_readings = np.zeros((3, n))
        
        # Extract USBL data
        usbl_meas = np.zeros((3, n))
        
        # Extract DVL data
        dvl_meas = np.zeros((3, n))
        
        # Extract depth data
        depth_meas = np.zeros(n)
        
        # Fill sensor data
        for i in range(1, n):
            if i < len(sensor_data):
                # IMU data - ensure data exists and is accessible
                if 'GyroReadings' in sensor_data[i].dtype.names and sensor_data[i]['GyroReadings'][0, 0].size > 0:
                    try:
                        # Get raw data
                        gyro_data = sensor_data[i]['GyroReadings'][0, 0]
                        # Check data shape
                        if gyro_data.size == 3 or len(gyro_data.shape) == 1 and gyro_data.shape[0] == 3:
                            # Normal 3D data
                            gyro_readings[:, i] = gyro_data.flatten()
                        elif len(gyro_data.shape) == 1 and gyro_data.shape[0] > 3:
                            # Abnormal shape data, take first 3 elements
                            print(f"Note: Gyro data {i} has abnormal shape ({gyro_data.shape}), taking first 3 elements")
                            gyro_readings[:, i] = gyro_data[:3]
                        elif len(gyro_data.shape) > 1:
                            
                            if len(gyro_data.shape) == 2 and gyro_data.shape[0] >= 1 and gyro_data.shape[1] >= 3:
                                
                                gyro_readings[:, i] = np.mean(gyro_data[:, :3], axis=0)
                            else:
                              
                                if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                                    gyro_readings[:, i] = gyro_readings[:, i-1]
                        else:
                            print(f"Warning: Cannot process gyro data {i}, shape: {gyro_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing gyro data {i}: {e}")
                        # Use previous valid value
                        if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                            gyro_readings[:, i] = gyro_readings[:, i-1]
                
                if 'AccelReadings' in sensor_data[i].dtype.names and sensor_data[i]['AccelReadings'][0, 0].size > 0:
                    try:
                        # Get raw data
                        accel_data = sensor_data[i]['AccelReadings'][0, 0]
                        # Check data shape
                        if accel_data.size == 3 or len(accel_data.shape) == 1 and accel_data.shape[0] == 3:
                            # Normal 3D data
                            accel_readings[:, i] = accel_data.flatten()
                        elif len(accel_data.shape) == 1 and accel_data.shape[0] > 3:
                            # Abnormal shape data, take first 3 elements
                            print(f"Note: Accelerometer data {i} has abnormal shape ({accel_data.shape}), taking first 3 elements")
                            accel_readings[:, i] = accel_data[:3]
                        elif len(accel_data.shape) > 1:
                            
                            if len(accel_data.shape) == 2 and accel_data.shape[0] >= 1 and accel_data.shape[1] >= 3:
                                # Average all sampling points to reduce noise
                                accel_readings[:, i] = np.mean(accel_data[:, :3], axis=0)
                            else:
                                # Use previous valid value for unprocessable cases
                                if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                                    accel_readings[:, i] = accel_readings[:, i-1]
                        else:
                            # Keep as zero for other abnormal cases
                            print(f"Warning: Cannot process accelerometer data {i}, shape: {accel_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing accelerometer data {i}: {e}")
                        # Use previous valid value
                        if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                            accel_readings[:, i] = accel_readings[:, i-1]
                
            
                if 'GNSSReading' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading'][0, 0].size > 0:
                    gnss = sensor_data[i]['GNSSReading'][0, 0].flatten()
                    if len(gnss) >= 3:
                        # Convert lat/lon to local coordinates
                        usbl_meas[0, i] = (gnss[1] - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
                        usbl_meas[1, i] = (gnss[0] - lat0) * np.pi * earth_radius / 180.0
                        usbl_meas[2, i] = gnss[2]  # Height
                
                # DVL data
                if 'DVL_Velocity' in sensor_data[i].dtype.names and sensor_data[i]['DVL_Velocity'][0, 0].size > 0:
                    try:
                        dvl_data = sensor_data[i]['DVL_Velocity'][0, 0]
                        # Check data shape
                        if dvl_data.size == 3 or len(dvl_data.shape) == 1 and dvl_data.shape[0] == 3:
                            dvl_meas[:, i] = dvl_data.flatten()
                        elif len(dvl_data.shape) == 1 and dvl_data.shape[0] > 3:
                            print(f"Note: DVL data {i} has abnormal shape ({dvl_data.shape}), taking first 3 elements")
                            dvl_meas[:, i] = dvl_data[:3]
                        else:
                            # Use previous valid value for unprocessable cases
                            if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                                dvl_meas[:, i] = dvl_meas[:, i-1]
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing DVL data {i}: {e}")
                        # Use previous valid value
                        if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                            dvl_meas[:, i] = dvl_meas[:, i-1]
                
             
                if 'GNSSReading_diff' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading_diff'][0, 0].size > 0:
                    gnss_diff = sensor_data[i]['GNSSReading_diff'][0, 0].flatten()
                    if len(gnss_diff) >= 3:
                        depth_meas[i] = gnss_diff[2]  # Depth value
        
        # Use real IMU data instead of simulation data
        true_gyro = gyro_readings
        true_accel = accel_readings
        gyro_err = np.zeros((n, 3))  # Bias will be considered in EKF
        accel_err = np.zeros((n, 3))  # Bias will be considered in EKF
        
        # Count valid data points
        valid_gyro_count = np.sum(np.any(true_gyro != 0, axis=0))
        valid_accel_count = np.sum(np.any(true_accel != 0, axis=0))
        valid_usbl_count = np.sum(np.any(usbl_meas != 0, axis=0))
        valid_dvl_count = np.sum(np.any(dvl_meas != 0, axis=0))
        valid_depth_count = np.sum(depth_meas != 0)
        
        print(f"Data loading complete, total {n} timestamps")
        print(f"Valid gyro data: {valid_gyro_count}/{n} ({valid_gyro_count/n*100:.1f}%)")
        print(f"Valid accelerometer data: {valid_accel_count}/{n} ({valid_accel_count/n*100:.1f}%)")
        print(f"Valid USBL data: {valid_usbl_count}/{n} ({valid_usbl_count/n*100:.1f}%)")
        print(f"Valid DVL data: {valid_dvl_count}/{n} ({valid_dvl_count/n*100:.1f}%)")
        print(f"Valid depth data: {valid_depth_count}/{n} ({valid_depth_count/n*100:.1f}%)")
        print(f"Trajectory start: {true_pos[:,0]}")
        print(f"Trajectory end: {true_pos[:,-1]}")
    
    # Initialize EKF
    ekf = EKF(params)
    
    # Initialize state estimation storage arrays
    est_pos = np.zeros((3, n))
    est_vel = np.zeros((3, n))
    est_att = np.zeros((3, n))
    
    # Set initial state
    if data_mode == 'simulation':
        # Use true trajectory initial state in simulation mode
        ekf.pos = true_pos[:, 0].copy()  # Use copy() to avoid reference passing
        ekf.vel = true_vel[:, 0].copy()
    else:
        # Use sensor data initial state in real data mode
        # Initial position
        initial_gnss = initial_state['GNSSReading'][0, 0].flatten()
        ekf.pos = np.array([
            (initial_gnss[1] - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0,
            (initial_gnss[0] - lat0) * np.pi * earth_radius / 180.0,
            initial_gnss[2]
        ])
        # Initial velocity
        if 'DVL_Velocity' in initial_state.dtype.names and initial_state['DVL_Velocity'][0, 0].size > 0:
            ekf.vel = initial_state['DVL_Velocity'][0, 0].flatten()
        else:
            print("Warning: No DVL_Velocity field in initial state, initializing with zero velocity")
            ekf.vel = np.zeros(3)  
        # Initial attitude
        if 'Orin_ref' in data_ref and data_ref['Orin_ref'].size > 0:
          
            print(f"Orin_ref shape: {data_ref['Orin_ref'].shape}")
            # Ensure correct attitude data extraction
            if data_ref['Orin_ref'].shape[1] >= 3:
                ekf.att = data_ref['Orin_ref'][0].copy()
            else:
                
                print("Warning: Cannot get complete attitude from reference data, using zero attitude")
                ekf.att = np.zeros(3)
    
    est_pos[:, 0] = ekf.pos
    est_vel[:, 0] = ekf.vel
    est_att[:, 0] = ekf.att
    
    # Main loop
    for i in range(1, n):
        # IMU prediction
        if data_mode == 'simulation':
            # Simulation mode
            gyro = true_gyro[:, i] + gyro_err[i]
            accel = true_accel[:, i] + accel_err[i]
            dt = params.dt
        else:
            # Real data mode
            # Check if IMU data is valid
            valid_gyro = not np.all(true_gyro[:, i] == 0) and true_gyro[:, i].shape == (3,)
            valid_accel = not np.all(true_accel[:, i] == 0) and true_accel[:, i].shape == (3,)
            
            if valid_gyro:
                gyro = true_gyro[:, i]
            else:
                # Use previous valid value or zero
                if i > 1 and not np.all(true_gyro[:, i-1] == 0):
                    gyro = true_gyro[:, i-1]
                    print(f"Note: Using gyro data {i-1} to replace invalid data {i}")
                else:
                    gyro = np.zeros(3)
                    print(f"Warning: Gyro data {i} invalid and no valid previous data found")
            
            if valid_accel:
                accel = true_accel[:, i]
            else:
                # Use previous valid value or zero
                if i > 1 and not np.all(true_accel[:, i-1] == 0):
                    accel = true_accel[:, i-1]
                    print(f"Note: Using accelerometer data {i-1} to replace invalid data {i}")
                else:
                    accel = np.zeros(3)
                    print(f"Warning: Accelerometer data {i} invalid and no valid previous data found")
            
            # Calculate actual time interval
            if i > 1:
                dt = t[i] - t[i-1]
            else:
                dt = 0.005  
        
        # EKF prediction step
        ekf.predict(dt, gyro, accel)
        
        # Sensor updates
        if data_mode == 'simulation':
            # Sensor update frequency in simulation mode
            if i % int(1.0/params.dt) == 0:  
                # USBL update
                ekf.update(usbl_meas[:, i], 'USBL')
                
            if i % int(0.1/params.dt) == 0: 
                # DVL update
                ekf.update(dvl_meas[:, i], 'DVL')
                ekf.update(np.array([depth_meas[i]]), 'Depth')
        else:
            # Sensor updates in real data mode
            # USBL update (check if USBL data is valid)
            if not np.all(usbl_meas[:, i] == 0):
                ekf.update(usbl_meas[:, i], 'USBL')
            
            # DVL update (check if DVL data is valid)
            if not np.all(dvl_meas[:, i] == 0):
                ekf.update(dvl_meas[:, i], 'DVL')
            
            # Depth update (check if depth data is valid)
            if depth_meas[i] != 0:
                ekf.update(np.array([depth_meas[i]]), 'Depth')
        
        # Record estimation results
        est_pos[:, i] = ekf.pos.copy()
        est_vel[:, i] = ekf.vel.copy()
        est_att[:, i] = ekf.att.copy()
        
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
    
    # *************EKF Simulation Loop**************
   
    # Visualize trajectory
    print(f"\nDisplaying trajectory using {data_mode} mode...")
    visualize_results(true_pos, est_pos, true_vel, est_vel, dvl_meas, depth_meas, t, data_mode)

def visualize_results(true_pos, est_pos, true_vel, est_vel, dvl_meas, depth_meas, t, data_mode='simulation'):
    """Visualize trajectory
    
    Display different types of trajectory plots based on data mode:
    - simulation mode: display 3D trajectory
    - real mode: display 2D planar trajectory using lat/lon coordinates with satellite map background
    """
    # Calculate total position error (for statistics)
    total_pos_error = np.linalg.norm(true_pos - est_pos, axis=0)
    
    if data_mode == 'simulation':
        # Simulation mode: display 3D trajectory
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, projection='3d')
        ax.plot(true_pos[0], true_pos[1], true_pos[2], 'b', linewidth=2, label='True Trajectory')
        ax.plot(est_pos[0], est_pos[1], est_pos[2], 'r--', linewidth=2, label='EKF Estimated Trajectory')
        
        ax.set_title('3D Trajectory Comparison', fontsize=14)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        ax.view_init(elev=30, azim=45)
        
    else:
        # Real world mode: display 2D planar trajectory using lat/lon coordinates with satellite map background
        try:
            import contextily as ctx
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.patheffects as PathEffects
            
            # Create figure window
            plt.figure(figsize=(14, 12))
            ax = plt.subplot(111)
            
            earth_radius = 6371000  # Earth radius in meters
            
            # Create custom color maps for trajectory gradient display
            colors_true = [(0, 0, 1), (0, 0, 0.5)]  # Blue gradient
            colors_est = [(1, 0, 0), (0.5, 0, 0)]   # Red gradient
            cmap_true = LinearSegmentedColormap.from_list('true_cmap', colors_true, N=100)
            cmap_est = LinearSegmentedColormap.from_list('est_cmap', colors_est, N=100)
            
            for i in range(len(t)-1):
                # True trajectory
                ax.plot(true_pos[0, i:i+2], true_pos[1, i:i+2], 'o-', 
                         color=cmap_true(i/len(t)), markersize=4, linewidth=2, alpha=0.8)
                # Estimated trajectory
                ax.plot(est_pos[0, i:i+2], est_pos[1, i:i+2], 'o-', 
                         color=cmap_est(i/len(t)), markersize=4, linewidth=2, alpha=0.8)
            
            # Mark start and end points
            ax.plot(true_pos[0, 0], true_pos[1, 0], 'go', markersize=10, label='Start')
            ax.plot(true_pos[0, -1], true_pos[1, -1], 'mo', markersize=10, label='End')
            
            txt1 = ax.text(true_pos[0, len(t)//2], true_pos[1, len(t)//2], 'True Trajectory', 
                          color='white', fontsize=12, fontweight='bold')
            txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='blue')])
            
            txt2 = ax.text(est_pos[0, len(t)//2], est_pos[1, len(t)//2], 'EKF Estimated Trajectory', 
                          color='white', fontsize=12, fontweight='bold')
            txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='red')])
            
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title('Underwater Vehicle Trajectory (EKF Navigation)', fontsize=14)
            
            # Add legend
            ax.legend(fontsize=12)
            
            try:
                # Get trajectory boundary range, slightly expanded for better display
                xmin, xmax = min(true_pos[0].min(), est_pos[0].min()), max(true_pos[0].max(), est_pos[0].max())
                ymin, ymax = min(true_pos[1].min(), est_pos[1].min()), max(true_pos[1].max(), est_pos[1].max())
                
                x_range = xmax - xmin
                y_range = ymax - ymin
                xmin -= 0.1 * x_range
                xmax += 0.1 * x_range
                ymin -= 0.1 * y_range
                ymax += 0.1 * y_range
                
                # Set display range
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                
                # Add satellite map background
                try:
                    ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)
                    print("Added terrain map as background")
                except Exception as e:
                    print(f"Error adding terrain map background: {e}")
                    print("Continuing with trajectory plot without background...")
                    
                print("Note: Map background is for reference only as local coordinates are used instead of actual lat/lon")
                print("For accurate satellite map, please ensure real lat/lon coordinates are used")
            except Exception as e:
                print(f"Error adding satellite map background: {e}")
                print("Continuing with trajectory plot without background...")
        
        except ImportError:
            print("Warning: contextily library not installed, cannot display satellite map background")
            print("Using regular 2D trajectory plot instead")
            
            # Create regular 2D trajectory plot
            plt.figure(figsize=(12, 10))
            plt.plot(true_pos[0], true_pos[1], 'b-', linewidth=2, label='True Trajectory')
            plt.plot(est_pos[0], est_pos[1], 'r--', linewidth=2, label='EKF Estimated Trajectory')
            plt.plot(true_pos[0, 0], true_pos[1, 0], 'go', markersize=10, label='Start')
            plt.plot(true_pos[0, -1], true_pos[1, -1], 'mo', markersize=10, label='End')
            
            plt.title('Underwater Vehicle Trajectory (EKF Navigation)', fontsize=14)
            plt.xlabel('X (m)', fontsize=12)
            plt.ylabel('Y (m)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nTrajectory Error Statistics:")
    print(f"Position RMS error: {np.sqrt(np.mean(total_pos_error**2)):.3f} m")
    print(f"Velocity RMS error: {np.sqrt(np.mean(np.sum((true_vel - est_vel)**2, axis=0))):.3f} m/s")
    
    # Save navigation results to CSV file
    print("\nSaving navigation results to CSV file...")
    
    # Prepare data: timestamps, true position(XYZ), estimated position(XYZ), true velocity(XYZ), estimated velocity(XYZ)
    csv_data = np.zeros((len(t), 13))  # 13 columns: time, true pos XYZ, est pos XYZ, true vel XYZ, est vel XYZ
    csv_data[:, 0] = t  # Timestamps
    csv_data[:, 1:4] = true_pos.T  # True position (transposed to match row format)
    csv_data[:, 4:7] = est_pos.T   # EKF estimated position (transposed to match row format)
    csv_data[:, 7:10] = true_vel.T  # True velocity
    csv_data[:, 10:13] = est_vel.T  # EKF estimated velocity
    
    # Set column headers
    header = "Timestamp,True Position X(m),True Position Y(m),True Position Z(m),EKF Estimated Position X(m),EKF Estimated Position Y(m),EKF Estimated Position Z(m),True Velocity X(m/s),True Velocity Y(m/s),True Velocity Z(m/s),EKF Estimated Velocity X(m/s),EKF Estimated Velocity Y(m/s),EKF Estimated Velocity Z(m/s)"
    
    # Choose filename based on data mode
    if data_mode == 'simulation':
        filename = "ekf_simulation_results.csv"
    else:
        filename = "ekf_real_world_results.csv"
    
    # Save to CSV file
    np.savetxt(filename, csv_data, delimiter=",", header=header, comments="")
    print(f"Navigation results saved to '{filename}'")


if __name__ == "__main__":
    main()