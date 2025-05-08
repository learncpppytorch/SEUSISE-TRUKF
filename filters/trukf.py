import numpy as np
import scipy.linalg
from scipy.linalg import expm, inv, cholesky, block_diag, svd, qr
from collections import deque
from attitude import euler_to_rotation_matrix, rotation_matrix_to_euler, skew, euler_to_quaternion, quaternion_to_euler, quaternion_multiply

# Nominal state container
class NominalState:
    def __init__(self):
        self.R = np.eye(3)      # SO(3) attitude matrix
        self.p = np.zeros(3)    # Global position 
        self.v = np.zeros(3)    # Global velocity
        self.bg = np.zeros(3)   # Gyroscope bias
        self.ba = np.zeros(3)   # Accelerometer bias

# Functions for converting between Lie algebra and rotation matrices
def so3_log(R):
    """Convert SO(3) rotation matrix to Lie algebra
    Input: R[3,3] - Rotation matrix
    Output: theta[3] - Rotation vector in Lie algebra
    """
    # Check for NaN or Inf in input
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("Warning: NaN or Inf detected in rotation matrix, returning zero vector")
        return np.zeros(3)
    
    # Ensure R is orthogonal
    # Use SVD decomposition to ensure orthogonality
    try:
        # Add condition number check
        if np.linalg.cond(R) > 1e10:
            print("Warning: Ill-conditioned rotation matrix, applying regularization")
            R = R + np.eye(3) * 1e-6
            
        U, S, Vh = np.linalg.svd(R)
        # Check if singular values are reasonable
        if np.any(S < 0.1) or np.any(S > 10):
            print(f"Warning: Unusual singular values in rotation matrix: {S}")
            S = np.clip(S, 0.1, 10)
            R = U @ np.diag(S) @ Vh
        else:
            R = U @ Vh
    except np.linalg.LinAlgError:
        # Use more robust symmetrization if SVD fails
        print("Warning: SVD failed in so3_log, using robust orthogonalization")
        R = (R + R.T) / 2
        try:
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.clip(eigvals, -1, 1)
            R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            print("Warning: Eigendecomposition failed, returning zero vector")
            return np.zeros(3)
    
    # Calculate rotation angle
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
    theta = np.arccos(cos_theta)
    
    # Special cases: theta close to 0 or π
    if np.abs(theta) < 1e-6:
        return np.zeros(3)  # No rotation
    elif np.abs(theta - np.pi) < 1e-6:
        # Handle 180 degree rotation
        if R[0,0] > -1 + 1e-6:
            return theta * np.array([1, 0, 0])
        elif R[1,1] > -1 + 1e-6:
            return theta * np.array([0, 1, 0])
        else:
            return theta * np.array([0, 0, 1])
    
    # General case
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])
    
    # Avoid division by near-zero values
    sin_theta = np.sin(theta)
    if np.abs(sin_theta) < 1e-6:
        sin_theta = 1e-6 * np.sign(sin_theta) if sin_theta != 0 else 1e-6
    
    # Check if axis vector is reasonable
    if np.any(np.isnan(axis)) or np.any(np.isinf(axis)):
        print("Warning: NaN or Inf detected in rotation axis, using fallback")
        # Use alternative method to calculate axis
        if np.abs(R[0,0] - 1) < 1e-6:
            axis = np.array([1, 0, 0])
        elif np.abs(R[1,1] - 1) < 1e-6:
            axis = np.array([0, 1, 0])
        else:
            axis = np.array([0, 0, 1])
    else:
        axis = axis / (2 * sin_theta)
    
    # Final check of result
    result = theta * axis
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("Warning: NaN or Inf in final so3_log result, returning zero vector")
        return np.zeros(3)
    
    return result

def so3_exp(theta):
    """Convert Lie algebra to SO(3) rotation matrix
    Input: theta[3] - Rotation vector in Lie algebra
    Output: R[3,3] - Rotation matrix
    """
    # Check for NaN or Inf in input
    if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
        print("Warning: NaN or Inf detected in rotation vector, returning identity")
        return np.eye(3)
    
    # Check input vector magnitude to prevent numerical overflow
    theta_norm = np.linalg.norm(theta)
    if theta_norm > 1e2:
        print(f"Warning: Very large rotation vector detected: {theta_norm}, clamping magnitude")
        theta = theta * (1e2 / theta_norm)
        theta_norm = 1e2
    
    # Special case: theta close to 0
    if theta_norm < 1e-6:
        return np.eye(3)
    
    # Calculate rotation axis and angle
    axis = theta / theta_norm
    theta_skew = skew(axis)
    
    # Use Rodrigues formula
    sin_theta = np.sin(theta_norm)
    cos_theta = np.cos(theta_norm)
    
    # Use more stable computation to avoid potential overflow in matrix multiplication
    try:
        R = np.eye(3) + sin_theta * theta_skew + (1 - cos_theta) * (theta_skew @ theta_skew)
    except Exception as e:
        print(f"Warning: Error in Rodrigues formula: {e}, using fallback")
        # Use alternative method
        R = expm(skew(theta))
    
    # Check if result contains NaN or Inf
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("Warning: NaN or Inf in rotation matrix, using fallback")
        try:
            # Use matrix exponential as fallback
            R = expm(skew(theta))
        except Exception:
            print("Warning: Fallback also failed, returning identity")
            return np.eye(3)
    
    # Ensure R is orthogonal using more robust SVD implementation
    try:
        # Check condition number
        if np.linalg.cond(R) > 1e10:
            print("Warning: Ill-conditioned matrix in so3_exp, applying regularization")
            R = R + np.eye(3) * 1e-6
        
        U, S, Vh = np.linalg.svd(R, full_matrices=False)
        # Check singular values
        if np.any(S < 0.1) or np.any(S > 10):
            print(f"Warning: Unusual singular values in so3_exp: {S}")
            S = np.clip(S, 0.1, 10)
            R = U @ np.diag(S) @ Vh
        else:
            R = U @ Vh
    except np.linalg.LinAlgError as e:
        print(f"Warning: SVD failed in so3_exp: {e}, using original matrix")
        # If SVD fails, try simple orthogonalization
        R = (R + R.T) / 2
        try:
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.clip(eigvals, 0.1, 10)
            R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            print("Warning: Eigendecomposition failed, returning original matrix")
    
    # Final check of result
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("Warning: NaN or Inf in final so3_exp result, returning identity")
        return np.eye(3)
    
    return R

# Transform matrix generation module
def build_T_matrix(nominal_state):
    """Construct transform matrix T(x)
    Input: nominal_state - Nominal state
    Output: T[15,15] - Transform matrix
    """
    T = np.eye(15)
    
    # Position skew-symmetric terms
    T[0:3, 3:6] = skew(nominal_state.p)
    
    # Velocity skew-symmetric terms
    T[3:6, 6:9] = skew(nominal_state.v)
    
    # Bias skew-symmetric terms
    T[6:9, 9:12] = skew(nominal_state.bg)
    T[9:12, 12:15] = skew(nominal_state.ba)
    
    return T

# Adaptive covariance adjustment module
class FadingWindow:
    def __init__(self, window_size=10):
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self, innovation, innovation_cov):
        """Update innovation sequence buffer
        Input:
            innovation - Innovation vector
            innovation_cov - Innovation covariance
        """
        normalized_innovation = innovation @ np.linalg.inv(innovation_cov) @ innovation
        self.buffer.append(normalized_innovation)
    
    def compute_lambda(self, threshold=1.5):
        """Calculate fading factor
        Output: lambda_k - Fading factor
        """
        if len(self.buffer) < self.window_size:
            return 1.0
        
        # Calculate mean of normalized innovation sequence
        mean_innovation = np.mean(self.buffer)
        
        # Calculate fading factor based on innovation sequence
        if mean_innovation > threshold:
            lambda_k = mean_innovation / threshold
        else:
            lambda_k = 1.0
            
        return lambda_k

# Huber loss weighting
def huber_weight(residual, S, delta=1.345):
    """Calculate Huber weight
    Input:
        residual - Residual vector
        S - Residual covariance
        delta - Huber threshold parameter
    Output:
        weight - Huber weight
    """
    # Calculate Mahalanobis distance
    mahalanobis = residual.T @ np.linalg.inv(S) @ residual
    
    # Apply Huber weight
    if mahalanobis <= delta**2:
        return 1.0
    else:
        return delta / np.sqrt(mahalanobis)

# Local filter interface
class LocalFilter:
    def __init__(self, sensor_type, params):
        self.sensor_type = sensor_type
        self.params = params
        self.last_update_time = 0
        
        # Initialize local state and covariance
        if sensor_type == 'DVL':
            self.state_dim = 3  # Velocity
            self.R = np.diag([params.dvl_noise**2] * 3)
        elif sensor_type == 'USBL':
            self.state_dim = 3  # Position
            self.R = np.diag(params.usbl_pos_error**2)
        elif sensor_type == 'Depth':
            self.state_dim = 1  # Depth
            self.R = np.array([[params.depth_bias**2]])
        
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10  # Initial covariance
    
    def dvl_projection(self, valid_beams, beam_directions):
        """DVL beam failure null space projection
        Input:
            valid_beams - Valid beam indices
            beam_directions - Beam direction matrix
        Output:
            projection_matrix - Projection matrix
        """
        if len(valid_beams) < 3:
            # Insufficient beams, use SVD for projection
            U, _, _ = svd(beam_directions[valid_beams])
            return U[:,:len(valid_beams)] @ np.linalg.pinv(U[:,:len(valid_beams)].T)
        else:
            # Sufficient beams, use full projection
            return np.eye(3)
    
    def usbl_delay_compensation(self, measurement, current_time, nominal_state):
        """USBL acoustic delay compensation
        Input:
            measurement - USBL measurement
            current_time - Current time
            nominal_state - Nominal state
        Output:
            compensated_measurement - Compensated measurement
        """
        # Calculate time delay
        delay = current_time - self.last_update_time
        
        # Compensate position using velocity
        compensated_measurement = measurement - nominal_state.v * delay
        
        return compensated_measurement
    
    def update(self, measurement, current_time, nominal_state=None):
       
        processed_measurement = measurement.copy()
        
        if self.sensor_type == 'USBL' and nominal_state is not None:
            
            processed_measurement = self.usbl_delay_compensation(
                measurement, current_time, nominal_state)
                      
            pos_data = np.abs(processed_measurement - nominal_state.p)
            large_mask = pos_data > 0.3
            if np.any(large_mask):
                measurement = processed_measurement - nominal_state.p
                measurement[large_mask] *= 0.1
                processed_measurement = nominal_state.p + measurement
            
            innovation = processed_measurement - self.state
            S = self.P + self.R
            K = self.P @ np.linalg.inv(S)
            self.state = self.state + K @ innovation
            self.P = (np.eye(self.state_dim) - K) @ self.P
            
        elif self.sensor_type == 'DVL':
            # Check for invalid beams
            if hasattr(measurement, 'beam_status'):
                valid_beams = np.where(measurement.beam_status > 0)[0]
                if len(valid_beams) < 4:
                    # Apply beam projection
                    beam_directions = np.array([  # Example beam directions
                        [0.866, 0, 0.5],
                        [-0.866, 0, 0.5],
                        [0, 0.866, 0.5],
                        [0, -0.866, 0.5]
                    ])
                    proj = self.dvl_projection(valid_beams, beam_directions)
                    processed_measurement = proj @ measurement
                    # Adjust noise covariance
                    self.R = proj @ self.R @ proj.T
                    # Update local state
                    innovation = processed_measurement - self.state
                    S = self.P + self.R
                    K = self.P @ np.linalg.inv(S)
                    self.state = self.state + K @ innovation
                    self.P = (np.eye(self.state_dim) - K) @ self.P
                    
        elif self.sensor_type == 'Depth':
            # Depth sensor only processes z-axis
            processed_measurement = np.array([measurement[0]])
            
            # Calculate depth error
            depth_error = np.abs(processed_measurement - self.state)
            
            # Update local state
            innovation = processed_measurement - self.state
            S = self.P + self.R
            K = self.P @ np.linalg.inv(S)
            self.state = self.state + K @ innovation
            self.P = (np.eye(self.state_dim) - K) @ self.P
        # Update timestamp
        self.last_update_time = current_time
        
        return processed_measurement, self.R

# Covariance intersection fusion
def covariance_intersection(P_list, x_list):
    """Covariance intersection fusion
    Input:
        P_list - List of covariance matrices
        x_list - List of state vectors
    Output:
        x_fused - Fused state
        P_fused - Fused covariance
    """
    n = len(P_list)
    if n == 0:
        return None, None
    elif n == 1:
        return x_list[0], P_list[0]
    
    # Calculate weights (based on trace of covariance matrices)
    weights = [1/(np.trace(P)+1e-6) for P in P_list]
    total_weight = sum(weights)
    weights = [w/total_weight for w in weights]
    
    # Fuse states
    dim = x_list[0].shape[0]
    P_inv_sum = np.zeros((dim, dim))
    P_inv_x_sum = np.zeros(dim)
    
    for i in range(n):
        # Use SVD decomposition for stable inversion
        try:
            U, s, Vh = np.linalg.svd(P_list[i])
            # Limit minimum singular values for numerical stability
            s = np.maximum(s, 1e-10)
            P_inv = (Vh.T * (1.0/s)) @ U.T
        except np.linalg.LinAlgError:
            # Use pseudoinverse if SVD fails
            P_inv = np.linalg.pinv(P_list[i])
            
        P_inv_sum += weights[i] * P_inv
        P_inv_x_sum += weights[i] * (P_inv @ x_list[i])
    
    # Use SVD decomposition for stable inversion
    try:
        U, s, Vh = np.linalg.svd(P_inv_sum)
        # Limit minimum singular values for numerical stability
        s = np.maximum(s, 1e-10)
        P_fused = (Vh.T * (1.0/s)) @ U.T
    except np.linalg.LinAlgError:
        # Use pseudoinverse if SVD fails
        P_fused = np.linalg.pinv(P_inv_sum)
        
    x_fused = P_fused @ P_inv_x_sum
    
    return x_fused, P_fused

# Federated Kalman filter fusion
def federated_fusion(P_list, x_list, beta_list=None):
    """Federated Kalman filter fusion
    Input:
        P_list - List of covariance matrices
        x_list - List of state vectors
        beta_list - List of information allocation coefficients
    Output:
        x_fused - Fused state
        P_fused - Fused covariance
    """
    n = len(P_list)
    if n == 0:
        return None, None
    elif n == 1:
        return x_list[0], P_list[0]
    
    # Default to uniform information allocation
    if beta_list is None:
        beta_list = [1.0/n] * n
    
    # Ensure beta_list and P_list have same length
    assert len(beta_list) == n, "Length of information allocation coefficient list must match length of covariance matrix list"
    
    # Fuse states
    dim = x_list[0].shape[0]
    P_inv_sum = np.zeros((dim, dim))
    P_inv_x_sum = np.zeros(dim)
    
    for i in range(n):
        # Use SVD decomposition for stable inversion
        try:
            U, s, Vh = np.linalg.svd(P_list[i])
            # Limit minimum singular values for numerical stability
            s = np.maximum(s, 1e-10)
            P_inv = (Vh.T * (1.0/s)) @ U.T
        except np.linalg.LinAlgError:
            # Use pseudoinverse if SVD fails
            P_inv = np.linalg.pinv(P_list[i])
            
        P_inv_sum += P_inv / beta_list[i]
        P_inv_x_sum += (P_inv @ x_list[i]) / beta_list[i]
    
    # Use SVD decomposition for stable inversion
    try:
        U, s, Vh = np.linalg.svd(P_inv_sum)
        # Limit minimum singular values for numerical stability
        s = np.maximum(s, 1e-10)
        P_fused = (Vh.T * (1.0/s)) @ U.T
    except np.linalg.LinAlgError:
        # Use pseudoinverse if SVD fails
        P_fused = np.linalg.pinv(P_inv_sum)
        
    x_fused = P_fused @ P_inv_x_sum
    
    return x_fused, P_fused

# Main filter class
class TRUKF:
    def __init__(self, params):
        # State dimension definitions
        self.pos_dim = 3  # Position dimension
        self.vel_dim = 3  # Velocity dimension
        self.att_dim = 3  # Attitude dimension
        self.bg_dim = 3   # Gyroscope bias dimension
        self.ba_dim = 3   # Accelerometer bias dimension
        self.total_dim = 15  # Total dimension
        
        # Initialize nominal state
        self.nominal_state = NominalState()
        
        # Initialize latest depth measurement
        self.latest_depth = None
        
        # Initialize error state and covariance
        self.delta_x = np.zeros(self.total_dim)
        self.P = np.diag([
            5**2, 5**2, 5**2,           # Position error covariance
            0.2**2, 0.2**2, 0.2**2,      # Velocity error covariance
            (0.1)**2, (0.1)**2, (0.2)**2,  # Attitude error covariance
            (0.01)**2, (0.01)**2, (0.01)**2,  # Gyroscope bias covariance
            (0.05)**2, (0.05)**2, (0.05)**2   # Accelerometer bias covariance
        ]).astype(np.float64)
        
        # Initialize square root covariance
        try:
            self.S = scipy.linalg.cholesky(self.P)
        except np.linalg.LinAlgError as e:
            print(f"Warning: Cholesky decomposition failed in initialization: {e}, using eigendecomposition")
            # Use eigenvalue decomposition as fallback
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure all eigenvalues are positive
            self.S = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
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
        
        # Initialize adaptive filter components
        self.fading_window = FadingWindow(window_size=10)
        
        # Initialize local filters
        self.local_filters = {
            'USBL': LocalFilter('USBL', params),
            'DVL': LocalFilter('DVL', params),
            'Depth': LocalFilter('Depth', params)
        }
        
        # Initialize sensor state dictionary (for federated fusion)
        self.sensor_states = {}
        self.last_fusion_time = 0
        
        # Current time
        self.current_time = 0
