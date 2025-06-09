import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

# Import custom modules
from params import EnhancedSensorParams
from trajectory.trajectory import generate_true_trajectory
from error_models import EnhancedErrorModels
from filters.trukf import TRUKF
from attitude import quaternion_to_euler

def main():
    # Initialize parameters
    params = EnhancedSensorParams()
    
    # Select data mode: 'simulation' or 'real'
    data_mode = 'simulation'
    
    if data_mode == 'simulation':
        # ===== Simulation Data Mode =====
        # Generate simulation trajectory
        true_pos, true_vel, true_gyro, true_accel, t = generate_true_trajectory(params)
        n = len(t)
        
        # Check trajectory start and end points
        print(f"Trajectory start point: {true_pos[:,0]}")
        print(f"Trajectory end point: {true_pos[:,-1]}")
        
        # Generate sensor data
        gyro_err, accel_err = EnhancedErrorModels.sins_error_model(
            np.zeros((n,3)), true_vel, params)
        
        # Extract true bias values (from parameters)
        true_gyro_bias = np.deg2rad(params.gyro_bias/3600)  # Convert units: deg/hour -> rad/sec
        true_accel_bias = params.accel_bias * 1e-3 * 9.8    # Convert units: mg -> m/s²
        
        # Generate observation data
        depth_meas = EnhancedErrorModels.depth_measurement(
            true_pos[2], 10+0.1*np.random.randn(n), params)
        dvl_meas = EnhancedErrorModels.dvl_measurement(
            true_vel, true_pos[2], t, params)
        usbl_meas = EnhancedErrorModels.usbl_measurement(true_pos, t, params)
        
        # Verify sensor data
        print(f"First 5 gyro errors: {gyro_err[:5]}")
        print(f"First 5 accel errors: {accel_err[:5]}")
        print(f"First 5 USBL measurements: {usbl_meas[:,:5]}")
        print(f"First 5 DVL measurements: {dvl_meas[:,:5]}")
        print(f"First 5 depth measurements: {depth_meas[:5]}")
    
    else:
        # ===== Real Data Mode =====
        print("Loading real data...")
        # Load reference data (true trajectory)
        data_ref = sio.loadmat('realworldData/sensorData_ref_9.mat')
        # Load sensor data
        data = sio.loadmat('realworldData/sensorData_9.mat')
        
        # Extract reference data (true trajectory)
        true_pos = np.zeros((3, data_ref['timestamps_ref'].shape[0]))
        # Convert lat/lon to local coordinates
        lat_ref = data_ref['lat_ref'].flatten()
        lon_ref = data_ref['lon_ref'].flatten()
  
        earth_radius = 6371000  # Earth radius in meters
        lat0, lon0 = lat_ref[0], lon_ref[0]  # Starting point as origin
        true_pos[0, :] = (lon_ref - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
        true_pos[1, :] = (lat_ref - lat0) * np.pi * earth_radius / 180.0
 
        print(f"att_ref shape: {data_ref['att_ref'].shape}")

        if 'GNSSReading_diff' in data['sensorData'][0][0].dtype.names:
  
            initial_depth = data['sensorData'][0][0]['GNSSReading_diff'][0, 0][0, 2] if data['sensorData'][0][0]['GNSSReading_diff'][0, 0].size > 0 else 0
            true_pos[2, :] = -initial_depth 
        else:
            true_pos[2, :] = 0  # Use 0 if no depth data available
        
        true_vel = data_ref['Velocity_ref'].T  # Transpose to match shape (3, n)
        
        # Extract timestamps
        t = data_ref['timestamps_ref'].flatten()
        n = len(t)
        
        sensor_data = data['sensorData'][0]
        
        # Initial state (first row of data)
        initial_state = sensor_data[0]
        
        gyro_readings = np.zeros((3, n))
        accel_readings = np.zeros((3, n))
        
        usbl_meas = np.zeros((3, n))
        
        dvl_meas = np.zeros((3, n))
        
        depth_meas = np.zeros(n)
        
        for i in range(1, n):
            if i < len(sensor_data):
         
                if 'GyroReadings' in sensor_data[i].dtype.names and sensor_data[i]['GyroReadings'][0, 0].size > 0:
                    try:
           
                        gyro_data = sensor_data[i]['GyroReadings'][0, 0]
         
                        if gyro_data.size == 3 or len(gyro_data.shape) == 1 and gyro_data.shape[0] == 3:
            
                            gyro_readings[:, i] = gyro_data.flatten()
                        elif len(gyro_data.shape) == 1 and gyro_data.shape[0] > 3:
      
                            print(f"Note: Gyro data shape abnormal at index {i} ({gyro_data.shape}), using first 3 elements only")
                            gyro_readings[:, i] = gyro_data[:3]
                        elif len(gyro_data.shape) > 1:
                            if len(gyro_data.shape) == 2 and gyro_data.shape[0] >= 1 and gyro_data.shape[1] >= 3:
              
                                gyro_readings[:, i] = np.mean(gyro_data[:, :3], axis=0)
                            else:
            
                                if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                                    gyro_readings[:, i] = gyro_readings[:, i-1]
                        else:
             
                            print(f"Warning: Unable to process gyro data at index {i}, shape: {gyro_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing gyro data at index {i}: {e}")
            
                        if i > 1 and not np.all(gyro_readings[:, i-1] == 0):
                            gyro_readings[:, i] = gyro_readings[:, i-1]
                
                if 'AccelReadings' in sensor_data[i].dtype.names and sensor_data[i]['AccelReadings'][0, 0].size > 0:
                    try:
                        accel_data = sensor_data[i]['AccelReadings'][0, 0]
            
                        if accel_data.size == 3 or len(accel_data.shape) == 1 and accel_data.shape[0] == 3:
                    
                            accel_readings[:, i] = accel_data.flatten()
                        elif len(accel_data.shape) == 1 and accel_data.shape[0] > 3:
                    
                            print(f"Note: Accelerometer data shape abnormal at index {i} ({accel_data.shape}), using first 3 elements only")
                            accel_readings[:, i] = accel_data[:3]
                        elif len(accel_data.shape) > 1:
                            if len(accel_data.shape) == 2 and accel_data.shape[0] >= 1 and accel_data.shape[1] >= 3:
             
                                accel_readings[:, i] = np.mean(accel_data[:, :3], axis=0)
                            else:
                     
                                if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                                    accel_readings[:, i] = accel_readings[:, i-1]
                        else:
            
                            print(f"Warning: Unable to process accelerometer data at index {i}, shape: {accel_data.shape}")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing accelerometer data at index {i}: {e}")
                   
                        if i > 1 and not np.all(accel_readings[:, i-1] == 0):
                            accel_readings[:, i] = accel_readings[:, i-1]
                
                if 'GNSSReading' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading'][0, 0].size > 0:
                    gnss = sensor_data[i]['GNSSReading'][0, 0].flatten()
                    if len(gnss) >= 3:
                        usbl_meas[0, i] = (gnss[1] - lon0) * np.cos(np.radians(lat0)) * np.pi * earth_radius / 180.0
                        usbl_meas[1, i] = (gnss[0] - lat0) * np.pi * earth_radius / 180.0
                        usbl_meas[2, i] = gnss[2]  
                
                # DVL data
                if 'DVL_Velocity' in sensor_data[i].dtype.names and sensor_data[i]['DVL_Velocity'][0, 0].size > 0:
                    try:
                        dvl_data = sensor_data[i]['DVL_Velocity'][0, 0]
               
                        if dvl_data.size == 3 or len(dvl_data.shape) == 1 and dvl_data.shape[0] == 3:
                            dvl_meas[:, i] = dvl_data.flatten()
                        elif len(dvl_data.shape) == 1 and dvl_data.shape[0] > 3:
                            print(f"Note: DVL data shape abnormal at index {i} ({dvl_data.shape}), using first 3 elements only")
                            dvl_meas[:, i] = dvl_data[:3]
                        else:
                            if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                                dvl_meas[:, i] = dvl_meas[:, i-1]
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing DVL data at index {i}: {e}")
         
                        if i > 1 and not np.all(dvl_meas[:, i-1] == 0):
                            dvl_meas[:, i] = dvl_meas[:, i-1]
     
                if 'GNSSReading_diff' in sensor_data[i].dtype.names and sensor_data[i]['GNSSReading_diff'][0, 0].size > 0:
                    gnss_diff = sensor_data[i]['GNSSReading_diff'][0, 0].flatten()
                    if len(gnss_diff) >= 3:
                        depth_meas[i] = gnss_diff[2] 
        
        # Use real IMU data instead of simulation data
        true_gyro = gyro_readings
        true_accel = accel_readings
        
        # Generate IMU errors (in real data mode, these errors are already included in measurements)
        gyro_err = np.zeros((n, 3))
        accel_err = np.zeros((n, 3))

    # *************TRUKF Simulation Loop**************
    # Initialize TRUKF
    kf = TRUKF(params)
    # Initialize storage variables
    TRUKF_pos = np.zeros_like(true_pos)

    # Set print interval (adjust based on data mode)
    print_interval = 500 if data_mode == 'simulation' else 500
    update_interval = int(1.0/params.dt) if data_mode == 'simulation' else 1

    for i in range(1, n):
        # 1. Get IMU measurements
        if data_mode == 'simulation':
            # Simulation mode: add errors to true values
            gyro_meas = true_gyro[:,i].copy() + gyro_err[i].copy()
            accel_meas = true_accel[:,i].copy() + accel_err[i].copy()
        else:
            # Real data mode: use measurements directly (errors included)
            gyro_meas = true_gyro[:,i].copy()
            accel_meas = true_accel[:,i].copy()

        # 2. TRUKF prediction step
        kf.predict(params.dt, gyro_meas.copy(), accel_meas.copy())
        
        # 3. Store current position estimate
        TRUKF_pos[:,i] = kf.get_position().copy()

        # 4. TRUKF update step
        if i % update_interval == 0:
      
            # Depth update priority
            if not np.isnan(depth_meas[i]):
                z_depth = np.array([depth_meas[i]]).copy()
                kf.update(z_depth, 'Depth', i)
                TRUKF_pos[:,i] = kf.get_position().copy()
            
            # USBL update
            if not np.any(np.isnan(usbl_meas[:,i])):
                z_usbl = usbl_meas[:,i].copy()
                kf.update(z_usbl, 'USBL', i)
                TRUKF_pos[:,i] = kf.get_position().copy()
                
            # DVL update
            if not np.any(np.isnan(dvl_meas[:,i])):
                z_dvl = dvl_meas[:,i].copy()
                kf.update(z_dvl, 'DVL', i)
                TRUKF_pos[:,i] = kf.get_position().copy()


        # Print key state variables
        if i % print_interval == 0:  # Adjust print frequency based on data mode
            print(f"Iter {i}:")
            print(f"    True Pos: {true_pos[:,i]}")
            print(f"    TRUKF Est Pos: {TRUKF_pos[:,i]}")
            print(f"    Gyro Bias: {kf.get_gyro_bias()}")
            print(f"    Accelerometer Bias: {kf.get_accel_bias()}")
            print(f"    P[0,0] (Position Error Covariance): {kf.P[0,0]}")
            print(f"Position Error: {np.linalg.norm(true_pos[:,i] - TRUKF_pos[:,i]):.3f} meters")
            # Print attitude angle information
            print(f"    Attitude Angles(rad): {kf.get_euler_angles()}")
            
            # Print Sigma point distribution (for debugging, only print first time in simulation mode)
            if i == print_interval and data_mode == 'simulation':  
                X = kf.generate_sigma_points()
                print("\nSigma Point Distribution Example (First 3 Points):")
                for j in range(3):
                    print(f"Sigma Point {j}: {X[j][:3]}")
                print(f"Mean: {kf.delta_x[:3]}")
                print(f"Covariance Diagonal: {np.diag(kf.P)[:3]}")
            
    # *************TRUKF Simulation Loop**************
   
    # Visualization and error analysis
    visualize_results(true_pos, TRUKF_pos, true_vel, dvl_meas, depth_meas, t, gyro_meas, accel_meas, usbl_meas, data_mode)

def visualize_results(true_pos, TRUKF_pos, true_vel, dvl_meas, depth_meas, t, gyro_meas, accel_meas, usbl_meas, data_mode='simulation'):
    """Visualization and Error Analysis"""
    # Create large figure window
    plt.figure(figsize=(20, 15))
    
    # 1. 3D trajectory comparison (top left)
    ax1 = plt.subplot(221, projection='3d')
    # True trajectory and TRUKF estimate
    ax1.plot(true_pos[0], true_pos[1], true_pos[2], 'b', linewidth=2, label='True Trajectory')
    ax1.plot(TRUKF_pos[0,:], TRUKF_pos[1,:], TRUKF_pos[2,:], 'r--', linewidth=2, label='TRUKF Estimate')
    
    ax1.set_title('3D Trajectory and Sensor Measurements', fontsize=12)
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # 2. Position error analysis (top right)
    ax2 = plt.subplot(222)
    pos_error_x = true_pos[0] - TRUKF_pos[0]
    pos_error_y = true_pos[1] - TRUKF_pos[1] 
    pos_error_z = true_pos[2] - TRUKF_pos[2]
    ax2.plot(t, pos_error_x, 'r', label='X-axis Error', linewidth=1.5)
    ax2.plot(t, pos_error_y, 'g', label='Y-axis Error', linewidth=1.5)
    ax2.plot(t, pos_error_z, 'b', label='Z-axis Error', linewidth=1.5)
    ax2.set_title('Position Error Analysis', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Error (m)', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    # 3. Choose different visualization content based on data mode
    ax3 = plt.subplot(223)
    if data_mode == 'simulation':
        # Simulation mode: show DVL velocity measurements
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
    else:
        # Real data mode: show sensor availability
        # Calculate sensor availability
        usbl_available = ~np.isnan(usbl_meas[0])
        dvl_available = ~np.isnan(dvl_meas[0])
        depth_available = ~np.isnan(depth_meas)
        
        # Create availability time series
        sensor_availability = np.zeros((3, len(t)))
        sensor_availability[0, usbl_available] = 1  # USBL available = 1, unavailable = 0
        sensor_availability[1, dvl_available] = 1   # DVL available = 1, unavailable = 0
        sensor_availability[2, depth_available] = 1 # Depth available = 1, unavailable = 0
        
        # Plot sensor availability
        ax3.plot(t, sensor_availability[0] + 0.1, 'r-', label='USBL Available', linewidth=1.5)
        ax3.plot(t, sensor_availability[1] + 0.0, 'g-', label='DVL Available', linewidth=1.5)
        ax3.plot(t, sensor_availability[2] - 0.1, 'b-', label='Depth Available', linewidth=1.5)
        ax3.set_title('Sensor Availability', fontsize=12)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Unavailable', 'Available'])
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Velocity (m/s)' if data_mode == 'simulation' else 'Availability', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True)
    
    # 4. Position error analysis (bottom right)
    ax4 = plt.subplot(224)
    # Calculate total position error
    total_pos_error = np.linalg.norm(true_pos - TRUKF_pos, axis=0)
    
    # Plot error
    ax4.plot(t, total_pos_error, 'r', label='TRUKF Position Error', linewidth=1.5)
    ax4.set_title('Position Error', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Error (m)', fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical information
    print("\nError Statistical Analysis:")
    print(f"TRUKF Position RMS Error: {np.sqrt(np.mean(total_pos_error**2)):.3f} m")
    
    # Save position data to CSV file
    print("\nSaving position data to CSV file...")
    # Prepare data: timestamps, true positions(XYZ), TRUKF estimated positions(XYZ)
    csv_data = np.zeros((len(t), 7))  # 7 columns: time, true X/Y/Z, estimated X/Y/Z
    csv_data[:, 0] = t  # timestamps
    csv_data[:, 1:4] = true_pos.T  # true positions (transposed to match row format)
    csv_data[:, 4:7] = TRUKF_pos.T   # TRUKF estimated positions (transposed to match row format)
    
    # Set column headers
    header = "Time(s),True Position X(m),True Position Y(m),True Position Z(m),TRUKF Estimated Position X(m),TRUKF Estimated Position Y(m),TRUKF Estimated Position Z(m)"
    
    # Set filename based on data mode
    filename = f"trukf_position_results_{data_mode}.csv"
    
    # Save to CSV file
    np.savetxt(filename, csv_data, delimiter=",", header=header, comments="")
    print(f"Position data has been saved to '{filename}'")

if __name__ == "__main__":
    main()