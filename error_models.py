import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter
from scipy.ndimage import shift

class EnhancedErrorModels:
    @staticmethod
    def sins_error_model(true_att, true_vel, params):
        n = len(true_att)
        
        # Gyroscope errors (fixed bias + random walk + first-order Markov) [7](@ref)
        gyro_bias = np.deg2rad(params.gyro_bias/3600)
        gyro_rw = np.deg2rad(params.gyro_rw/60)*np.sqrt(params.dt)*np.random.randn(n,3)
       
        # Use first-order Markov process to simulate gyroscope drift
        gyro_markov = np.zeros((n,3))
        for i in range(1,n):
            gyro_markov[i] = 0.9*gyro_markov[i-1] + 0.1*np.random.randn(3)
        gyro_err = gyro_bias + gyro_rw + 0.005*gyro_markov  # Reduce Markov process influence coefficient
        
        # Accelerometer errors (fixed bias + Markov process) [3](@ref)
        accel_bias = params.accel_bias * 1e-3 * 9.8
        accel_markov = np.zeros((n,3))
        for i in range(1,n):
            accel_markov[i] = 0.95*accel_markov[i-1] + 0.05*np.random.randn(3)
        accel_err = accel_bias + 0.005*accel_markov
        
        return gyro_err, accel_err
    
    @staticmethod
    def dvl_measurement(true_vel, depth, t, params):
        """DVL measurement model (including mounting pole vibration)"""
        n = true_vel.shape[1]

        # Generate 3D vibration angles (different frequencies around XYZ axes)
        vib_angle = np.array([
            # X-axis vibration: reduced amplitude, 5Hz
            0.05 * np.sin(2 * np.pi * 5 * t),
            # Y-axis vibration: reduced amplitude, 3Hz
            0.03 * np.sin(2 * np.pi * 3 * t + np.pi/4),
            # Z-axis vibration: reduced amplitude, 2Hz
            0.04 * np.sin(2 * np.pi * 2 * t + np.pi/2)
        ])  # Unit: degrees
    
        # Convert vibration angles to radians and add installation bias
        dynamic_angle = params.dvl_install_angle[:, None] + np.deg2rad(vib_angle)
    
        # Batch calculate rotation matrices
        vel_body = np.zeros_like(true_vel)
        for i in range(n):
        # Construct current skew-symmetric matrix
            so3_matrix = np.array([
                [0, -dynamic_angle[2,i], dynamic_angle[1,i]],
                [dynamic_angle[2,i], 0, -dynamic_angle[0,i]],
                [-dynamic_angle[1,i], dynamic_angle[0,i], 0]
            ])
            # Calculate rotation matrix using matrix exponential
            R = expm(so3_matrix)
            vel_body[:,i] = R @ true_vel[:,i]
    
        # Bottom-lock state judgment [9](@ref) When vehicle is far from seafloor, DVL cannot obtain bottom-tracking data, vertical velocity measurement error increases
        if np.mean(depth) > params.dvl_height_threshold:
            # Water-layer tracking vertical velocity noise increases (proportional to depth), reduce noise coefficient
            depth_factor = np.clip(depth / params.dvl_height_threshold, 1, 2)
            vel_body[2,:] += 0.05 * depth_factor * np.random.randn(n)
    
        # Composite error model
        scale_error = 1 + params.dvl_scale * np.random.randn(n)
        temp_effect = 1 + 0.001 * (t - t.mean())  # Temperature drift term
        return temp_effect * scale_error * vel_body + params.dvl_noise * np.random.randn(*vel_body.shape)

    @staticmethod
    def depth_measurement(true_depth, temp, params):
        # Nonlinear error (quadratic term)
        nonlinear = params.depth_nonlinear * true_depth**2 * 1e-4
        # Temperature drift
        temp_effect = params.depth_temp_drift * (temp - 25)
        # Use fixed noise standard deviation
        return true_depth + params.depth_bias + nonlinear + temp_effect + 0.01*np.random.randn()

    @staticmethod 
    def usbl_measurement(true_pos, t, params):
        """Improved USBL (including non-integer time shift interpolation)"""
        
        # Time synchronization error modeling (supports non-integer time shift)
        time_shift = params.usbl_time_sync_error / params.dt
    
        # 3D position interpolation processing (considering independent axis characteristics)
        shifted_pos = np.empty_like(true_pos)
        for i in range(3):  # Process X/Y/Z axes separately
            # Use cubic spline interpolation (mode='nearest' to prevent boundary anomalies)
            shifted_pos[i] = shift(true_pos[i], 
                                shift=time_shift,
                                mode='nearest',
                                order=3)  # Cubic spline interpolation
        
        # Add errors only to XY axes, keep Z axis true value
        pos_error = np.zeros((3, len(t)))
        pos_error[0:2] = params.usbl_pos_error[0:2, None] * np.random.randn(1, len(t))
        
        # Add temperature drift only to XY axes
        temp_drift = np.zeros((3, len(t)))
        temp_drift[0:2] = 0.005 * (t - t.mean()) * np.random.randn(1, len(t))
        
        return shifted_pos + pos_error + temp_drift
