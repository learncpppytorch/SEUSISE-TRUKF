import numpy as np
from scipy.linalg import expm, inv, cholesky
from attitude import euler_to_rotation_matrix, rotation_matrix_to_euler, skew, euler_to_quaternion, quaternion_to_euler, quaternion_multiply

class UKF:
    def __init__(self, params):
        # State dimension definitions
        self.pos_dim = 3  # Position dimension
        self.vel_dim = 3  # Velocity dimension 
        self.att_dim = 3  # Attitude dimension
        self.bg_dim = 3   # Gyroscope bias dimension
        self.ba_dim = 3   # Accelerometer bias dimension
        self.total_dim = 15  # Total dimension
        
        # State initialization
        self.pos_nominal = np.zeros(3)  # Position
        self.vel_nominal = np.zeros(3)  # Velocity
        self.att_nominal = np.zeros(3)  # Attitude (Euler angles)
        self.bg_nominal = np.zeros(3)   # Gyroscope bias
        self.ba_nominal = np.zeros(3)   # Accelerometer bias
        
        # State vector
        self.x = np.zeros(self.total_dim)
        self.x[:3] = self.pos_nominal
        self.x[3:6] = self.vel_nominal
        self.x[6:9] = self.att_nominal
        self.x[9:12] = self.bg_nominal
        self.x[12:15] = self.ba_nominal
        
        # Initialize covariance matrix
        self.P = np.diag([
            5**2, 5**2, 10**2,           # Position error covariance
            0.2**2, 0.2**2, 0.5**2,      # Velocity error covariance
            (0.1)**2, (0.1)**2, (0.2)**2,  # Attitude error covariance
            (0.01)**2, (0.01)**2, (0.01)**2,  # Gyroscope bias covariance
            (0.05)**2, (0.05)**2, (0.05)**2   # Accelerometer bias covariance
        ]).astype(np.float64)
        
        # Process noise matrix Q
        self.Q = np.diag([
            0.05**2, 0.05**2, 0.1**2,        # Position process noise
            0.01**2, 0.01**2, 0.02**2,       # Velocity process noise
            0.001**2, 0.001**2, 0.002**2,    # Attitude process noise
            (0.0001)**2, (0.0001)**2, (0.0001)**2,  # Gyroscope bias process noise
            (0.001)**2, (0.001)**2, (0.001)**2      # Accelerometer bias process noise
        ]).astype(np.float64)
        
        # Measurement noise matrices
        self.R_usbl = np.diag(params.usbl_pos_error**2)
        self.R_dvl = np.diag([params.dvl_noise**2] * 3)
        self.R_depth = np.array([[params.depth_bias**2]])
        
        # Gravity constant
        self.g = np.array([0, 0, -9.81])
        
        # Navigation result storage
        self.nav_pos = np.zeros((3, int(params.total_time/params.dt)+1))
        
        # UKF parameters
        self.n = self.total_dim  # State dimension
        self.alpha = 0.1  # Scaling parameter, typically small positive
        self.beta = 2.0   # Prior distribution parameter, optimal for Gaussian
        self.kappa = 0.0  # Secondary scaling parameter, typically 0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Calculate weights
        self.compute_weights()

    def compute_weights(self):
        """Calculate Sigma point weights"""
        # Number of Sigma points
        self.n_sigma = 2 * self.n + 1
        
        # Mean weights
        self.Wm = np.zeros(self.n_sigma)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wm[1:] = 1.0 / (2 * (self.n + self.lambda_))
        
        # Covariance weights
        self.Wc = np.zeros(self.n_sigma)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = self.Wm[1:]

    def generate_sigma_points(self):
        """Generate Sigma points"""
        # Ensure covariance matrix is positive definite
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry
        
        # Add regularization term for numerical stability
        min_eig = np.min(np.linalg.eigvals(self.P))
        if min_eig < 1e-10:
            self.P += np.eye(self.n) * (1e-10 - min_eig)
        
        try:
            # Calculate matrix square root
            L = cholesky((self.n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure all eigenvalues are positive
            L = eigvecs @ np.diag(np.sqrt((self.n + self.lambda_) * eigvals)) @ eigvecs.T
        
        # Generate Sigma points
        X = np.zeros((self.n_sigma, self.n))
        X[0] = self.x
        for i in range(self.n):
            X[i+1] = self.x + L[i]
            X[i+1+self.n] = self.x - L[i]
        
        return X

    def predict(self, dt, gyro, accel):
        """Prediction step"""
        # 1. Generate Sigma points
        X = self.generate_sigma_points()
        
        # 2. Propagate Sigma points
        X_pred = np.zeros_like(X)
        for i in range(self.n_sigma):
            X_pred[i] = self.propagate_state(X[i], dt, gyro, accel)
        
        # 3. Calculate predicted mean
        self.x = np.zeros(self.n)
        for i in range(self.n_sigma):
            self.x += self.Wm[i] * X_pred[i]
        
        # 4. Calculate predicted covariance
        self.P = np.zeros((self.n, self.n))
        for i in range(self.n_sigma):
            diff = X_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        
        # 5. Add process noise
        self.P += self.Q * dt
        
        # 6. Update nominal states
        self.pos_nominal = self.x[:3]
        self.vel_nominal = self.x[3:6]
        self.att_nominal = self.x[6:9]
        self.bg_nominal = self.x[9:12]
        self.ba_nominal = self.x[12:15]

    def propagate_state(self, x, dt, gyro, accel):
        """Nonlinear state propagation model"""
        # Decompose state vector
        pos = x[:3]
        vel = x[3:6]
        att = x[6:9]
        bg = x[9:12]
        ba = x[12:15]
        
        # Compensate IMU biases
        gyro_corrected = gyro - bg
        accel_corrected = accel - ba
        
        # Attitude update
        R_old = euler_to_rotation_matrix(att)
        omega_skew = skew(gyro_corrected)
        dR = expm(omega_skew * dt)
        R_new = R_old @ dR
        att_new = rotation_matrix_to_euler(R_new)
        
        # Velocity update
        accel_nav = R_new.T @ accel_corrected + self.g
        vel_new = vel + accel_nav * dt
        
        # Position update
        pos_new = pos + vel_new * dt
        
        # Bias update (assumed random walk)
        bg_new = bg  # Gyroscope bias remains constant
        ba_new = ba  # Accelerometer bias remains constant
        
        # Combine new state
        x_new = np.zeros_like(x)
        x_new[:3] = pos_new
        x_new[3:6] = vel_new
        x_new[6:9] = att_new
        x_new[9:12] = bg_new
        x_new[12:15] = ba_new
        
        return x_new

    def update(self, z, sensor_type, i=None, return_nis=False):
        """Measurement update step
        
        Args:
            sensor_type: Sensor type ('USBL', 'DVL', 'Depth')
            i: Time step index (optional)
            return_nis: Whether to return Normalized Innovation Squared (NIS)
            
        Returns:
            nis: Normalized Innovation Squared value (if return_nis is True)
        """
        # Create copy of input data to avoid modifying original
        z = z.copy()
        
        # Add data validity check
        if sensor_type == 'USBL':
            # USBL data anomaly detection
            if np.any(np.abs(z) > 1000):  
                print("USBL data anomaly, skipping update")
                return None if return_nis else None
                
        elif sensor_type == 'DVL':
            # DVL data anomaly detection
            if np.any(np.abs(z) > 100):    
                print("DVL data anomaly, skipping update")
                return None if return_nis else None
        
        # 1. Generate Sigma points
        X = self.generate_sigma_points()
        
        # 2. Transform Sigma points to measurement space
        Z = np.zeros((self.n_sigma, self.get_measurement_dim(sensor_type)))
        for i in range(self.n_sigma):
            Z[i] = self.measurement_model(X[i], sensor_type)
        
        # 3. Calculate predicted measurement mean
        z_pred = np.zeros(self.get_measurement_dim(sensor_type))
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * Z[i]
        
        # 4. Calculate measurement covariance and cross-covariance
        Pzz = np.zeros((self.get_measurement_dim(sensor_type), self.get_measurement_dim(sensor_type)))
        Pxz = np.zeros((self.n, self.get_measurement_dim(sensor_type)))
        
        for i in range(self.n_sigma):
            diff_z = Z[i] - z_pred
            diff_x = X[i] - self.x
            Pzz += self.Wc[i] * np.outer(diff_z, diff_z)
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # 5. Add measurement noise
        R = self.get_measurement_noise(sensor_type)
        Pzz += R
        
        # 6. Calculate Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # 7. Calculate innovation
        innovation = z - z_pred
        
        # Calculate Normalized Innovation Squared (NIS)
        nis = None
        if return_nis:
            nis = innovation.T @ np.linalg.inv(Pzz) @ innovation
        
        # 8. Update state and covariance
        self.x = self.x + K @ innovation
        self.P = self.P - K @ Pzz @ K.T
        
        # Ensure covariance matrix symmetry
        self.P = (self.P + self.P.T) / 2
        
        # 9. Update state components
        self.pos_nominal = self.x[:3]
        self.vel_nominal = self.x[3:6]
        self.att_nominal = self.x[6:9]
        self.bg_nominal = self.x[9:12]
        self.ba_nominal = self.x[12:15]
        
        # 10. Store navigation results
        if i is not None:
            self.nav_pos[:,i] = self.pos_nominal
            
        # Return NIS value
        if return_nis:
            return nis

    def measurement_model(self, x, sensor_type):
        """Nonlinear measurement model"""
        if sensor_type == 'USBL':
            # USBL directly measures position
            return x[:3]
            
        elif sensor_type == 'DVL':
            # DVL directly measures velocity
            return x[3:6]
            
        elif sensor_type == 'Depth':
            # Depth sensor only measures Z-axis position
            return np.array([x[2]])
        
        return None

    def get_measurement_dim(self, sensor_type):
        """Get measurement dimension"""
        if sensor_type == 'USBL':
            return 3
        elif sensor_type == 'DVL':
            return 3
        elif sensor_type == 'Depth':
            return 1
        return 0

    def get_measurement_noise(self, sensor_type):
        """Get measurement noise matrix"""
        if sensor_type == 'USBL':
            return self.R_usbl
        elif sensor_type == 'DVL':
            return self.R_dvl
        elif sensor_type == 'Depth':
            return self.R_depth
        return None
        
    def sensor_fault_detection(self, measurement, sensor_type, threshold=5):
        """Sensor fault detection
        
        Args:
            measurement: Sensor measurement
            sensor_type: Sensor type
            threshold: Threshold
            
        Returns:
            bool: Whether sensor data is valid
        """
        if sensor_type == 'DVL':
            # Check DVL bottom tracking status
            if measurement.get('beam_status', np.array([0,0,0,0])).sum() < 3:
                print("DVL bottom tracking failed, skipping update")
                return False
                
            # Check if DVL velocity is abnormal
            if np.any(np.abs(measurement.get('velocity', np.zeros(3))) > threshold):
                print("DVL velocity abnormal, skipping update")
                return False
                
        elif sensor_type == 'USBL':
            # Check if USBL position is abnormal
            if np.any(np.abs(measurement) > 1000):
                print("USBL position abnormal, skipping update")
                return False
                
        return True
        
    def reset(self):
        """Reset filter state"""
        self.pos_nominal = np.zeros(3)
        self.vel_nominal = np.zeros(3)
        self.att_nominal = np.zeros(3)
        self.bg_nominal = np.zeros(3)
        self.ba_nominal = np.zeros(3)
        
        self.x = np.zeros(self.total_dim)
        self.x[:3] = self.pos_nominal
        self.x[3:6] = self.vel_nominal
        self.x[6:9] = self.att_nominal
        self.x[9:12] = self.bg_nominal
        self.x[12:15] = self.ba_nominal
        
        # Reset covariance matrix
        self.P = np.diag([
            5**2, 5**2, 10**2,           # Position error covariance
            0.2**2, 0.2**2, 0.5**2,      # Velocity error covariance
            (0.1)**2, (0.1)**2, (0.2)**2,  # Attitude error covariance
            (0.01)**2, (0.01)**2, (0.01)**2,  # Gyroscope bias covariance
            (0.05)**2, (0.05)**2, (0.05)**2   # Accelerometer bias covariance
        ]).astype(np.float64)