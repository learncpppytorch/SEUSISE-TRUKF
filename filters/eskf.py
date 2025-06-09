import numpy as np
from scipy.linalg import expm, inv

def euler_to_quaternion(euler):
    """Convert Euler angles to quaternion
    Input: euler[3] - [roll, pitch, yaw]
    Output: quaternion[4] - [w, x, y, z]
    """
    roll, pitch, yaw = euler
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles
    Input: quaternion[4] - [w, x, y, z]
    Output: euler[3] - [roll, pitch, yaw]
    """
    w, x, y, z = q
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def quaternion_multiply(q1, q2):
    """Quaternion multiplication"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def euler_to_rotation_matrix(euler):
    """Convert Euler angles to rotation matrix
    Input: euler[3] - [roll, pitch, yaw]
    Output: R[3,3] - Rotation matrix from navigation frame to body frame
    """
    roll, pitch, yaw = euler
    
    # Calculate sine and cosine of the angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Build rotation matrix
    R = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
        [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]
    ])
    
    return R

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles
    Input: R[3,3] - Rotation matrix from navigation frame to body frame
    Output: euler[3] - [roll, pitch, yaw]
    """
    # Extract pitch angle (y-axis rotation)
    pitch = -np.arcsin(R[0,2])
    
    # Extract roll angle (x-axis rotation)
    roll = np.arctan2(R[1,2], R[2,2])
    
    # Extract yaw angle (z-axis rotation)
    yaw = np.arctan2(R[0,1], R[0,0])
    
    return np.array([roll, pitch, yaw])

def skew(v):
    """Calculate skew-symmetric matrix of vector
    Input: v[3] - 3D vector
    Output: S[3,3] - Skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

class ESKF:
    def __init__(self, params):
        # State dimensions
        self.pos_dim = 3  # Position dimension
        self.vel_dim = 3  # Velocity dimension
        self.att_dim = 3  # Attitude error dimension
        self.bg_dim = 3   # Gyroscope bias dimension
        self.ba_dim = 3   # Accelerometer bias dimension
        self.total_dim = 15  # Total dimension
        
        # Initialize nominal states
        self.pos_nominal = np.zeros(3)  # Position
        self.vel_nominal = np.zeros(3)  # Velocity
        self.att_nominal = np.zeros(3)  # Attitude (Euler angles)
        self.bg_nominal = np.zeros(3)   # Gyroscope bias
        self.ba_nominal = np.zeros(3)   # Accelerometer bias
        
        # Initialize error state
        self.delta_x = np.zeros(self.total_dim)
        
        # Initialize covariance matrix
        self.P = np.diag([
            5**2, 5**2, 10**2,           # Position error covariance: reduced initial value
            0.2**2, 0.2**2, 0.5**2,      # Velocity error covariance: reduced initial value
            (0.1)**2, (0.1)**2, (0.2)**2,  # Attitude error covariance: unchanged
            (0.01)**2, (0.01)**2, (0.01)**2,  # Gyroscope bias covariance
            (0.05)**2, (0.05)**2, (0.05)**2   # Accelerometer bias covariance
        ]).astype(np.float64)
        
        # Process noise matrix Q
        self.Q = np.diag([
            0.05**2, 0.05**2, 0.1**2,        # Position process noise: reduced noise
            0.01**2, 0.01**2, 0.02**2,       # Velocity process noise: reduced noise
            0.001**2, 0.001**2, 0.002**2,    # Attitude process noise: unchanged
            (0.0001)**2, (0.0001)**2, (0.0001)**2,  # Gyroscope bias process noise: reduced noise
            (0.001)**2, (0.001)**2, (0.001)**2      # Accelerometer bias process noise: reduced noise
        ]).astype(np.float64)
        
        # Measurement noise matrices
        self.R_usbl = np.diag(params.usbl_pos_error**2)
        self.R_dvl = np.diag([params.dvl_noise**2] * 3)
        self.R_depth = np.array([[params.depth_bias**2]])
        
        # Gravity constant
        self.g = np.array([0, 0, -9.81])

    def predict(self, dt, gyro, accel):
        """Prediction step"""
        # 1. Compensate IMU biases
        gyro_corrected = gyro - self.bg_nominal
        accel_corrected = accel - self.ba_nominal
        
        # 2. Update nominal states
        # Attitude update
        R_old = euler_to_rotation_matrix(self.att_nominal)
        omega_skew = skew(gyro_corrected)
        # Update rotation matrix using matrix exponential
        dR = expm(omega_skew * dt)
        R_new = R_old @ dR  # Modified: adjusted rotation matrix multiplication order
        self.att_nominal = rotation_matrix_to_euler(R_new)
        
        # Velocity update - corrected to navigation frame
        accel_nav = R_new.T @ accel_corrected + self.g  # Convert from body to navigation frame
        
        self.vel_nominal += accel_nav * dt
                
        # Position update
        self.pos_nominal += self.vel_nominal * dt
        
        # 3. Calculate error state transition matrix
        F = self._compute_error_state_transition(dt, R_new, accel_corrected)
        
        # 4. Predict error state covariance
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z, sensor_type, return_nis=False):
        """Measurement update step
        
        Parameters:
            z: measurement
            sensor_type: sensor type ('USBL', 'DVL', 'Depth')
            return_nis: whether to return Normalized Innovation Squared (NIS)
            
        Returns:
            nis: Normalized Innovation Squared value (if return_nis is True)
        """
        # Add data validity check
        if sensor_type == 'USBL':
            # USBL data anomaly detection
            if np.any(np.abs(z) > 10000):  # Position jump detection
                print("USBL data anomaly, skipping update")
                return None if return_nis else None
                
        elif sensor_type == 'DVL':
            # DVL data anomaly detection
            if np.any(np.abs(z) > 100):    # Velocity anomaly detection
                print("DVL data anomaly, skipping update")
                return None if return_nis else None
            
        # Build observation matrix and noise matrix
        H, R, z_pred = self._get_observation_matrix(sensor_type)
        
        # Calculate innovation
        innovation = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Ensure numerical stability
        S = (S + S.T) / 2  # Ensure symmetry
        
        # Calculate Normalized Innovation Squared (NIS)
        nis = None
        if return_nis:
            nis = innovation.T @ np.linalg.inv(S) @ innovation
        
        # Calculate Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state
        self.delta_x = K @ innovation
        
        # Update error covariance (Joseph form)
        I = np.eye(self.total_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        
        # Inject error state into nominal state
        self._inject_error_state()
        
        # Reset error state and update covariance
        G = np.eye(self.total_dim)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * self.delta_x[6:9])
        self.P = G @ self.P @ G.T
        self.delta_x = np.zeros(self.total_dim)
        
        # Return NIS value if requested
        if return_nis:
            return nis

    def _compute_error_state_transition(self, dt, R, accel):
        """Calculate error state transition matrix"""
        F = np.eye(self.total_dim)
        
        # Position derivative with respect to velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity derivative with respect to attitude
        F[3:6, 6:9] = -R @ skew(accel) * dt
        
        # Velocity derivative with respect to accelerometer bias
        F[3:6, 12:15] = -R * dt  # Modified: use correct coordinate transformation
        
        # Attitude derivative with respect to gyroscope bias
        F[6:9, 9:12] = -np.eye(3) * dt
        
        return F

    def _get_observation_matrix(self, sensor_type):
        """Build observation matrix"""
        if sensor_type == 'USBL':
            H = np.zeros((3, self.total_dim))
            H[:, 0:3] = np.eye(3)
            R = self.R_usbl
            z_pred = self.pos_nominal
            
        elif sensor_type == 'DVL':
            H = np.zeros((3, self.total_dim))
            H[:, 3:6] = np.eye(3)
            R = self.R_dvl
            z_pred = self.vel_nominal
            
        elif sensor_type == 'Depth':
            H = np.zeros((1, self.total_dim))
            H[0, 2] = 1.0
            R = self.R_depth
            z_pred = np.array([self.pos_nominal[2]])
            
        return H, R, z_pred

    def _inject_error_state(self):
        """Inject error state into nominal state"""
        # Add position and velocity directly
        self.pos_nominal += self.delta_x[0:3]
        self.vel_nominal += self.delta_x[3:6]
        
        # Use SO(3) exponential mapping for attitude
        dR = expm(skew(self.delta_x[6:9]))
        R = dR @ euler_to_rotation_matrix(self.att_nominal)
        self.att_nominal = rotation_matrix_to_euler(R)
        
        # Add biases directly
        self.bg_nominal += self.delta_x[9:12]
        self.ba_nominal += self.delta_x[12:15]
