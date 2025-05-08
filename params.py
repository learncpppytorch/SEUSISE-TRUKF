import numpy as np

class EnhancedSensorParams:
    def __init__(self):
        # SINS error parameters (with time-varying characteristics)
        self.gyro_bias = np.array([0.1, 0.15, -0.08])  # Fixed bias (deg/hour)
        self.gyro_rw = np.array([0.01, 0.015, 0.008])  # Random walk (deg/√h)
        self.accel_bias = np.array([0.02, 0.03, 0.01])  # Fixed bias (mg)
        self.accel_markov = np.array([0.005, 0.007, 0.003])  # First-order Markov process (mg)
        self.init_att_error = np.deg2rad([0.5, 0.5, 1])  # Initial attitude error
        
        # DVL error model
        self.dvl_scale = 5e-4  # Velocity scale error, reduced by half
        self.dvl_noise = 0.05  # White noise (m/s), reduced by half
        self.dvl_install_angle = np.deg2rad([0.3, -0.2, 0.8])  # Installation bias angle [3](@ref), reduced installation bias
        self.dvl_height_threshold = 150  # Bottom-lock height threshold (m)[9](@ref)
        
        # USBL error model
        self.usbl_pos_error = np.array([0.3, 0.3, 0.3])  # Position error standard deviation (m), reduced error
        self.usbl_time_sync_error = 0.05  # Time synchronization error (s)[4](@ref), reduced sync error
        self.usbl_update_interval = 5  # seconds
        
        # Depth sensor error model [6](@ref)
        self.depth_bias = 0.05  # Static bias (m)
        self.depth_nonlinear = 0.02 # Nonlinear error (%FS)
        self.depth_temp_drift = 0.001  # Temperature drift (°C^-1)
        
        # Environmental disturbance parameters [1](@ref)
        self.current_speed = 0.1  # Ocean current velocity (m/s)
        self.current_dir = np.deg2rad(45)  # Ocean current direction
        self.wave_amplitude = 0.5  # Wave disturbance amplitude (m)
        
        # Time parameters
        self.dt = 0.1  # Sampling time
        self.total_time = 600  # Total duration (seconds)
