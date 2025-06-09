import numpy as np
import scipy.linalg
from scipy.linalg import expm, inv, cholesky, block_diag, svd, qr
from collections import deque
from attitude import euler_to_rotation_matrix, rotation_matrix_to_euler, skew, euler_to_quaternion, quaternion_to_euler, quaternion_multiply

# Nominal State Container
class NominalState:
    def __init__(self):
        self.R = np.eye(3)      # SO(3) attitude matrix
        self.p = np.zeros(3)    # Global position
        self.v = np.zeros(3)    # Global velocity  
        self.bg = np.zeros(3)   # Gyroscope bias
        self.ba = np.zeros(3)   # Accelerometer bias

# Lie Algebra and Rotation Matrix Conversion Functions
def so3_log(R):
    """Convert SO(3) rotation matrix to Lie algebra
    Input: R[3,3] - rotation matrix
    Output: theta[3] - rotation vector in Lie algebra
    """
    # Check if input contains NaN or Inf
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("Warning: NaN or Inf detected in rotation matrix, returning zero vector")
        return np.zeros(3)
    
    # Ensure R is orthogonal matrix
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
        # If SVD fails, use more robust orthogonalization
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
    Input: theta[3] - rotation vector in Lie algebra
    Output: R[3,3] - rotation matrix
    """
    # Check if input contains NaN or Inf
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
        R = expm(skew(theta))
    
    # Check result
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("Warning: NaN or Inf in rotation matrix, using fallback")
        try:
            R = expm(skew(theta))
        except Exception:
            print("Warning: Fallback also failed, returning identity")
            return np.eye(3)
    
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

# Transformation Matrix Generation Module
def build_T_matrix(nominal_state):
    """Construct transformation matrix T(x)
    Input: nominal_state - nominal state
    Output: T[15,15] - transformation matrix
    """
    T = np.eye(15)
    
    # Position skew terms
    T[0:3, 3:6] = skew(nominal_state.p)
    
    # Velocity skew terms  
    T[3:6, 6:9] = skew(nominal_state.v)
    
    # Bias skew terms
    T[6:9, 9:12] = skew(nominal_state.bg)
    T[9:12, 12:15] = skew(nominal_state.ba)
    
    return T

# Adaptive Covariance Adjustment Module
class FadingWindow:
    def __init__(self, window_size=10):
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self, innovation, innovation_cov):
        """Update innovation sequence buffer
        Input:
            innovation - innovation vector
            innovation_cov - innovation covariance
        """
        normalized_innovation = innovation @ np.linalg.inv(innovation_cov) @ innovation
        self.buffer.append(normalized_innovation)
    
    def compute_lambda(self, threshold=1.5):
        """Calculate fading factor
        Output: lambda_k - fading factor
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

# Huber Loss Weighting
def huber_weight(residual, S, delta=1.345):
    """Calculate Huber weight
    Input:
        residual - residual vector
        S - residual covariance
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

# Local Filter Interface
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
            valid_beams - valid beam indices
            beam_directions - beam direction matrix
        Output:
            projection_matrix - projection matrix
        """
        if len(valid_beams) < 3:
            # Insufficient beams, use SVD decomposition for projection
            U, _, _ = svd(beam_directions[valid_beams])
            return U[:,:len(valid_beams)] @ np.linalg.pinv(U[:,:len(valid_beams)].T)
        else:
            # Sufficient beams, use full projection
            return np.eye(3)
    
    def usbl_delay_compensation(self, measurement, current_time, nominal_state):
        """USBL acoustic delay compensation
        Input:
            measurement - USBL measurement
            current_time - current time
            nominal_state - nominal state
        Output:
            compensated_measurement - compensated measurement
        """
        # Calculate time delay
        delay = current_time - self.last_update_time
        
        # Compensate position using velocity
        compensated_measurement = measurement - nominal_state.v * delay
        
        return compensated_measurement
    
    def update(self, measurement, current_time, nominal_state=None):
        """Local filter update
        Input:
            measurement - sensor measurement
            current_time - current time
            nominal_state - nominal state (for delay compensation)
        Output:
            processed_measurement - processed measurement
            R - measurement noise covariance
        """
        # Apply sensor-specific processing
        processed_measurement = measurement.copy()
        
        if self.sensor_type == 'USBL' and nominal_state is not None:
            # USBL delay compensation
            processed_measurement = self.usbl_delay_compensation(
                measurement, current_time, nominal_state)
            
            # Calculate position
            pos_data = np.abs(processed_measurement - nominal_state.p)
            large_mask = pos_data > 0.3
            if np.any(large_mask):
                measurement = processed_measurement - nominal_state.p
                measurement[large_mask] *= 0.1
                processed_measurement = nominal_state.p + measurement
            
            # Update local state
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

# Covariance Intersection Fusion
def covariance_intersection(P_list, x_list):
    """Covariance intersection fusion
    Input:
        P_list - covariance matrix list
        x_list - state vector list
    Output:
        x_fused - fused state
        P_fused - fused covariance
    """
    n = len(P_list)
    if n == 0:
        return None, None
    elif n == 1:
        return x_list[0], P_list[0]
    
    # Calculate weights
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
            # If SVD fails, use pseudoinverse
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
        P_fused = np.linalg.pinv(P_inv_sum)
        
    x_fused = P_fused @ P_inv_x_sum
    
    return x_fused, P_fused

# Federated Kalman Filter Fusion
def federated_fusion(P_list, x_list, beta_list=None):
    """Federated Kalman filter fusion
    Input:
        P_list - covariance matrix list
        x_list - state vector list
        beta_list - information allocation coefficient list (optional)
    Output:
        x_fused - fused state
        P_fused - fused covariance
    """
    n = len(P_list)
    if n == 0:
        return None, None
    elif n == 1:
        return x_list[0], P_list[0]
    
    # Default uniform information allocation
    if beta_list is None:
        beta_list = [1.0/n] * n
    
    # Ensure beta_list and P_list have same length
    assert len(beta_list) == n, "Information allocation coefficient list length must match covariance matrix list length"
    
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
            # If SVD fails, use pseudoinverse
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
        # If SVD fails, use pseudoinverse
        P_fused = np.linalg.pinv(P_inv_sum)
        
    x_fused = P_fused @ P_inv_x_sum
    
    return x_fused, P_fused

# Main Filter Class
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
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-10)
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
        self.alpha = 0.1  # Scaling parameter
        self.beta = 2.0   # Prior distribution parameter
        self.kappa = 0.0  # Secondary scaling parameter
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
        
        # Initialize sensor state dictionary
        self.sensor_states = {}
        self.last_fusion_time = 0
        
        # Current time
        self.current_time = 0
    
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
        """在变换空间生成Sigma点"""
        # 检查当前状态是否包含NaN或Inf
        if np.any(np.isnan(self.delta_x)) or np.any(np.isinf(self.delta_x)):
            print("Warning: NaN or Inf detected in error state, resetting to zero")
            self.delta_x = np.zeros(self.n)
            
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            print("Warning: NaN or Inf detected in covariance matrix, resetting")
            self.P = np.diag([
                5**2, 5**2, 5**2,           # 位置误差协方差
                0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
            ]).astype(np.float64)
        
        # 构建变换矩阵
        T = build_T_matrix(self.nominal_state)
        
        # 检查变换矩阵的条件数
        try:
            cond_T = np.linalg.cond(T)
            if cond_T > 1e10:
                print(f"Warning: Ill-conditioned transformation matrix: {cond_T}, using regularized version")
                # 添加正则化
                T = T + np.eye(self.n) * 1e-6
        except Exception as e:
            print(f"Warning: Error checking condition number: {e}")
        
        # 变换误差状态和协方差到η空间
        try:
            eta = T @ self.delta_x
            
            # 检查结果
            if np.any(np.isnan(eta)) or np.any(np.isinf(eta)):
                print("Warning: NaN or Inf after transformation, using original error state")
                eta = self.delta_x.copy()
        except Exception as e:
            print(f"Warning: Error in state transformation: {e}, using original error state")
            eta = self.delta_x.copy()
        
        # 添加额外的数值稳定性检查
        try:
            # 使用更稳健的方式计算变换后的协方差
            try:
                # 直接计算，但确保结果是实数
                P_eta = np.real(T @ self.P @ T.T)
            except Exception as e:
                print(f"Warning: Direct computation failed: {e}, using block computation")
                # 分块计算以减少数值误差
                P_eta = np.zeros_like(self.P)
                block_size = 5  # 每次处理5列
                for i in range(0, self.n, block_size):
                    end = min(i + block_size, self.n)
                    P_block = T @ self.P[:, i:end]
                    P_eta[:, i:end] = P_block
                P_eta = np.real(T @ P_eta.T)  # 确保结果是实数
                P_eta = P_eta.T  # 转置回来
            
            # 检查结果
            if np.any(np.isnan(P_eta)) or np.any(np.isinf(P_eta)):
                print("Warning: NaN or Inf in transformed covariance, using diagonal approximation")
                P_eta = np.diag(np.diag(self.P))
            
            # 确保协方差矩阵对称性
            P_eta = (P_eta + P_eta.T) / 2
            
            # 添加强正则化项
            P_eta += np.eye(self.n) * 1e-3  # 增加正则化强度
            
            # 确保协方差矩阵对称性
            P_eta = (P_eta + P_eta.T) / 2
            
            # 检查特征值，确保正定性
            try:
                eigvals = np.linalg.eigvalsh(P_eta)
                min_eig = np.min(eigvals)
                if min_eig < 1e-4:                     
                    P_eta += np.eye(self.n) * (1e-4 - min_eig)
            except Exception as e:
                print(f"Warning: Eigenvalue computation failed: {e}, adding default regularization")
                P_eta += np.eye(self.n) * 1e-4
            
            # 尝试Cholesky分解
            try:
                # 确保使用实数部分
                scaled_P = np.real((self.n + self.lambda_) * P_eta)
                L = cholesky(scaled_P)
            except np.linalg.LinAlgError as e:
                print(f"Warning: Cholesky decomposition failed: {e}, using eigendecomposition")
                # 直接使用特征值分解，避免SVD可能的复数问题
                try:
                    eigvals, eigvecs = np.linalg.eigh(P_eta)  # eigh保证实对称矩阵的特征值分解
                    eigvals = np.maximum(eigvals, 1e-4)  # 确保所有特征值为正且足够大
                    # 使用实数部分计算平方根
                    L = np.real(eigvecs @ np.diag(np.sqrt((self.n + self.lambda_) * eigvals)))
                except Exception as e:
                    print(f"Warning: Eigendecomposition failed: {e}, using diagonal approximation")
                    # 使用对角近似
                    diag_vals = np.maximum(np.diag(P_eta), 1e-4)  # 确保对角元素为正且足够大
                    L = np.diag(np.sqrt((self.n + self.lambda_) * diag_vals))
        except Exception as e:
            # 如果出现任何错误，使用简单的对角矩阵
            print(f"Warning: Using diagonal matrix for sigma points: {e}")
            # 确保对角元素为正
            diag_P = np.maximum(np.diag(self.P), 1e-4)  # 提高最小值阈值
            L = np.diag(np.sqrt((self.n + self.lambda_) * diag_P))
        
        # 生成Sigma点
        X = np.zeros((self.n_sigma, self.n))
        X[0] = eta
        for i in range(self.n):
            if i < L.shape[0]:  # 确保索引有效
                X[i+1] = eta + L[i]
                X[i+1+self.n] = eta - L[i]
            else:
                # 如果L的维度不足，使用零向量
                X[i+1] = eta
                X[i+1+self.n] = eta
        
        # 检查生成的Sigma点
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or Inf in sigma points, using simplified generation")
            # 使用简化的Sigma点生成
            X = np.zeros((self.n_sigma, self.n))
            X[0] = self.delta_x
            for i in range(self.n):
                X[i+1] = self.delta_x.copy()
                X[i+1][i] += 0.01  # 添加小扰动
                X[i+1+self.n] = self.delta_x.copy()
                X[i+1+self.n][i] -= 0.01  # 添加小扰动
            return X
        
        # 变换回原始空间
        try:
            # 检查T的条件数
            cond_T = np.linalg.cond(T)
            if cond_T > 1e10:
                print(f"Warning: Ill-conditioned for inversion: {cond_T}, using pseudoinverse")
                T_inv = np.linalg.pinv(T)
            else:
                T_inv = np.linalg.inv(T)
                
            # 检查逆矩阵
            if np.any(np.isnan(T_inv)) or np.any(np.isinf(T_inv)):
                print("Warning: NaN or Inf in inverse matrix, using pseudoinverse")
                T_inv = np.linalg.pinv(T)
                
            # 逐点变换，检查每个结果
            for i in range(self.n_sigma):
                X_transformed = np.real(T_inv @ X[i])  # 确保结果是实数
                if np.any(np.isnan(X_transformed)) or np.any(np.isinf(X_transformed)):
                    print(f"Warning: NaN or Inf after inverse transformation for point {i}, using original point")
                    X[i] = self.delta_x.copy()
                else:
                    X[i] = X_transformed
        except np.linalg.LinAlgError as e:
            # 如果矩阵求逆失败，使用伪逆
            print(f"Warning: Matrix inversion failed: {e}, using pseudoinverse")
            try:
                T_inv = np.linalg.pinv(T)
                for i in range(self.n_sigma):
                    X[i] = np.real(T_inv @ X[i])  # 确保结果是实数
            except Exception as e:
                print(f"Warning: Pseudoinverse also failed: {e}, using original error state")
                for i in range(self.n_sigma):
                    X[i] = self.delta_x.copy()
        except Exception as e:
            # 如果出现其他错误，使用原始误差状态
            print(f"Warning: Unexpected error in inverse transformation: {e}, using original error state")
            for i in range(self.n_sigma):
                X[i] = self.delta_x.copy()
        
        # 最终检查结果
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or Inf in final sigma points, using simplified set")
            # 使用简化的Sigma点集
            X = np.zeros((self.n_sigma, self.n))
            X[0] = self.delta_x
            for i in range(self.n):
                X[i+1] = self.delta_x.copy()
                X[i+1][i] += 0.1
                X[i+1+self.n] = self.delta_x.copy()
                X[i+1+self.n][i] -= 0.1
        
        return X
    
    def predict(self, dt, gyro, accel):
        """预测步骤"""
      
        if np.any(np.isnan(gyro)) or np.any(np.isinf(gyro)):
            print("Warning: NaN or Inf detected in gyro input, replacing with zeros")
            gyro = np.zeros(3)
            
        if np.any(np.isnan(accel)) or np.any(np.isinf(accel)):
            print("Warning: NaN or Inf detected in accel input, replacing with zeros")
            accel = np.zeros(3)
        
        # 更新当前时间
        self.current_time += dt
        
        # 1. 名义状态传播
        try:
            self.propagate_nominal_state(dt, gyro, accel)
        except Exception as e:
            print(f"Warning: Error in nominal state propagation: {e}, skipping")
        
        # 2. 生成Sigma点 
        try:
            # 使用简化的Sigma点生成方法，避免数值不稳定性
            X = np.zeros((self.n_sigma, self.n))
            X[0] = self.delta_x.copy()
            
            # 计算协方差的平方根（使用对角近似）
            diag_P = np.maximum(np.diag(self.P), 1e-4)  # 确保对角元素为正且足够大
            sqrt_diag = np.sqrt((self.n + self.lambda_) * diag_P)
            
            # 生成Sigma点
            for i in range(self.n):
                X[i+1] = self.delta_x.copy()
                X[i+1][i] += sqrt_diag[i]  
                X[i+1+self.n] = self.delta_x.copy()
                X[i+1+self.n][i] -= sqrt_diag[i]  
        except Exception as e:
            print(f"Warning: Error in sigma point generation: {e}, using very simplified generation")
       
            X = np.zeros((self.n_sigma, self.n))
            X[0] = self.delta_x.copy()
            for i in range(self.n):
                X[i+1] = self.delta_x.copy()
                X[i+1][i] += 0.1  
                X[i+1+self.n] = self.delta_x.copy()
                X[i+1+self.n][i] -= 0.1  
        
        # 3. 传播Sigma点
        X_pred = np.zeros_like(X)
        for i in range(self.n_sigma):
            try:
                X_pred[i] = self.propagate_error_state(X[i], dt, gyro, accel)
                
                if np.any(np.isnan(X_pred[i])) or np.any(np.isinf(X_pred[i])):
                    print(f"Warning: NaN or Inf in propagated sigma point {i}, using original point")
                    X_pred[i] = X[i].copy()
            except Exception as e:
                print(f"Warning: Error propagating sigma point {i}: {e}, using original point")
                X_pred[i] = X[i].copy()
        
        # 4. 计算预测均值
        try:
            self.delta_x = np.zeros(self.n)
            for i in range(self.n_sigma):
                self.delta_x += self.Wm[i] * X_pred[i]
                
            if np.any(np.isnan(self.delta_x)) or np.any(np.isinf(self.delta_x)):
                print("Warning: NaN or Inf in mean calculation, resetting to zero")
                self.delta_x = np.zeros(self.n)
        except Exception as e:
            print(f"Warning: Error in mean calculation: {e}, resetting to zero")
            self.delta_x = np.zeros(self.n)
        
        # 5. 计算预测协方差
        try:
            self.P = np.diag([
                2.5**2, 2.5**2, 5**2,           # 位置误差协方差
                0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
            ]).astype(np.float64)
            
            # 逐点更新协方差
            for i in range(self.n_sigma):
                diff = X_pred[i] - self.delta_x
                
                # 检查差值
                if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
                    print(f"Warning: NaN or Inf in sigma point difference {i}, skipping")
                    continue
                
                diff_norm = np.linalg.norm(diff)
                if diff_norm > 2:
                    diff = diff * (2 / diff_norm)
                
                try:
                    outer_prod = np.outer(diff, diff)
                    
                    # 检查外积结果
                    if np.any(np.isnan(outer_prod)) or np.any(np.isinf(outer_prod)):
                        print(f"Warning: NaN or Inf in outer product {i}, skipping")
                        continue
                    
                    max_val = np.max(np.abs(outer_prod))
                    if max_val > 1e3:
                        print(f"Warning: Large outer product value: {max_val}, scaling down")
                        outer_prod = outer_prod * (1e3 / max_val)
                    
                    # 应用权重并累加
                    self.P += self.Wc[i] * outer_prod
                except Exception as e:
                    print(f"Warning: Error in outer product calculation: {e}, skipping")
                    continue
            
            # 确保协方差矩阵对称性和正定性
            self.P = (self.P + self.P.T) / 2
            
            # 检查协方差矩阵的特征值
            try:
                eigvals = np.linalg.eigvalsh(self.P)
                min_eig = np.min(eigvals)
                if min_eig < 1e-4:
                    self.P += np.eye(self.n) * (1e-4 - min_eig)
            except Exception as e:
                print(f"Warning: Error in eigenvalue computation: {e}, adding default regularization")
                self.P += np.eye(self.n) * 1e-4
        except Exception as e:
            print(f"Warning: Error in covariance calculation: {e}, using diagonal approximation")
            # 使用对角近似
            self.P = np.diag([
                5**2, 5**2, 5**2,           # 位置误差协方差
                0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
            ]).astype(np.float64)
        
        # 6. 添加过程噪声
        try:
            # 限制过程噪声的大小
            Q_scaled = self.Q.copy() * dt
            max_Q = np.max(np.diag(Q_scaled))
            if max_Q > 1.0:
                print(f"Warning: Large process noise: {max_Q}, scaling down")
                Q_scaled = Q_scaled * (1.0 / max_Q)
            
            # 添加过程噪声
            self.P += Q_scaled
            
            # 检查结果
            if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
                print("Warning: NaN or Inf after adding process noise, using diagonal approximation")
                # 使用对角近似
                self.P = np.diag([
                    5**2, 5**2, 5**2,           # 位置误差协方差
                    0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                    (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                    (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                    (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
                ]).astype(np.float64)
        except Exception as e:
            print(f"Warning: Error adding process noise: {e}, using diagonal approximation")
            # 使用对角近似
            self.P = np.diag([
                5**2, 5**2, 5**2,           # 位置误差协方差
                0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
            ]).astype(np.float64)
        
        # 7. 更新平方根协方差（使用QR分解）
        try:
            self.P = (self.P + self.P.T) / 2
            
            try:
                eigvals = np.linalg.eigvalsh(self.P)
                min_eig = np.min(eigvals)
                if min_eig < 1e-6:
                    self.P += np.eye(self.n) * (1e-6 - min_eig)
            except Exception as e:
                print(f"Warning: Eigenvalue computation failed: {e}, adding regularization")
                self.P += np.eye(self.n) * 1e-6
            
            # 构建增广矩阵
            n = self.n
            A = np.zeros((n, n + n))
            A[:, :n] = self.P
            
            # 使用Cholesky分解计算过程噪声的平方根
            try:
                Q_sqrt = scipy.linalg.cholesky(self.Q)
                A[:, n:] = np.sqrt(dt) * Q_sqrt
            except np.linalg.LinAlgError as e:
                print(f"Warning: Cholesky decomposition failed for Q: {e}, using diagonal approximation")
                # 使用对角近似
                Q_diag = np.sqrt(np.diag(self.Q))
                A[:, n:] = np.sqrt(dt) * np.diag(Q_diag)
            
            # QR分解
            try:
                Q, R = qr(A.T, mode='economic')
                self.S = R.T
                
                if np.any(np.isnan(self.S)) or np.any(np.isinf(self.S)):
                    print("Warning: NaN or Inf in square root covariance, using Cholesky")
                    # 使用Cholesky分解作为备选方案
                    self.S = scipy.linalg.cholesky(self.P)
            except Exception as e:
                print(f"Warning: QR decomposition failed: {e}, using Cholesky")
       
                try:
                    self.S = scipy.linalg.cholesky(self.P)
                except np.linalg.LinAlgError:
                    print("Warning: Cholesky also failed, using diagonal approximation")
              
                    self.S = np.diag(np.sqrt(np.diag(self.P)))
            
            # 8. 确保协方差矩阵对称性和正定性
            try:
                self.P = self.S @ self.S.T
                
                # 检查结果
                if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
                    print("Warning: NaN or Inf in reconstructed covariance, using diagonal approximation")
                    # 使用对角近似
                    self.P = np.diag([
                        5**2, 5**2, 5**2,           # 位置误差协方差
                        0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                        (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                        (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                        (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
                    ]).astype(np.float64)
                    self.S = np.diag(np.sqrt(np.diag(self.P)))
                
                # 确保对称性和正定性
                self.P = (self.P + self.P.T) / 2 + 1e-8 * np.eye(n)
            except Exception as e:
                print(f"Warning: Error reconstructing covariance: {e}, using diagonal approximation")
                # 使用对角近似
                self.P = np.diag([
                    5**2, 5**2, 5**2,           # 位置误差协方差
                    0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                    (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                    (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                    (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
                ]).astype(np.float64)
                self.S = np.diag(np.sqrt(np.diag(self.P)))
        except Exception as e:
            print(f"Warning: Error in square root update: {e}, using diagonal approximation")
            # 使用对角近似
            self.P = np.diag([
                5**2, 5**2, 5**2,           # 位置误差协方差
                0.2**2, 0.2**2, 0.5**2,      # 速度误差协方差
                (0.1)**2, (0.1)**2, (0.2)**2,  # 姿态误差协方差
                (0.01)**2, (0.01)**2, (0.01)**2,  # 陀螺仪偏置协方差
                (0.05)**2, (0.05)**2, (0.05)**2   # 加速度计偏置协方差
            ]).astype(np.float64)
            self.S = np.diag(np.sqrt(np.diag(self.P)))
    
    def propagate_nominal_state(self, dt, gyro, accel):
        """传播名义状态"""
        # 检查输入数据
        if np.any(np.isnan(gyro)) or np.any(np.isinf(gyro)):
            print("Warning: NaN or Inf detected in gyro input, replacing with zeros")
            gyro = np.zeros(3)
            
        if np.any(np.isnan(accel)) or np.any(np.isinf(accel)):
            print("Warning: NaN or Inf detected in accel input, replacing with zeros")
            accel = np.zeros(3)
            
        # 补偿IMU偏置
        gyro_corrected = gyro - self.nominal_state.bg
        accel_corrected = accel - self.nominal_state.ba
        
        if np.any(np.isnan(gyro_corrected)) or np.any(np.isinf(gyro_corrected)):
            print("Warning: NaN or Inf in corrected gyro data, using zeros")
            gyro_corrected = np.zeros(3)
            
        if np.any(np.isnan(accel_corrected)) or np.any(np.isinf(accel_corrected)):
            print("Warning: NaN or Inf in corrected accel data, using zeros")
            accel_corrected = np.zeros(3)
        
        try:
            omega_skew = skew(gyro_corrected)
            dR = expm(omega_skew * dt)
            
            # 检查旋转矩阵增量
            if np.any(np.isnan(dR)) or np.any(np.isinf(dR)):
                print("Warning: NaN or Inf in rotation increment, using identity")
                dR = np.eye(3)
                
            # 更新旋转矩阵
            R_new = self.nominal_state.R @ dR
            
            # 检查更新后的旋转矩阵
            if np.any(np.isnan(R_new)) or np.any(np.isinf(R_new)):
                print("Warning: NaN or Inf in updated rotation matrix, keeping previous")
            else:
                # 确保旋转矩阵正交性
                U, S, Vh = np.linalg.svd(R_new)
                self.nominal_state.R = U @ Vh
        except Exception as e:
            print(f"Warning: Error in attitude update: {e}, keeping previous attitude")
        
        try:
            # 速度更新
            accel_nav = self.nominal_state.R.T @ accel_corrected + self.g
            
            # 检查导航系下的加速度
            if np.any(np.isnan(accel_nav)) or np.any(np.isinf(accel_nav)):
                print("Warning: NaN or Inf in navigation frame acceleration, using gravity only")
                accel_nav = self.g
                
            accel_norm = np.linalg.norm(accel_nav)
            if accel_norm > 15: 
                print(f"Warning: Large acceleration detected: {accel_norm}, clamping magnitude")
                accel_nav = accel_nav * (15 / accel_norm)
                
            # 更新速度
            v_new = self.nominal_state.v + accel_nav * dt
            
            if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
                print("Warning: NaN or Inf in updated velocity, keeping previous")
            else:
                v_norm = np.linalg.norm(v_new)
                if v_norm > 2:  
                    v_new = v_new * (2 / v_norm)
                self.nominal_state.v = v_new
        except Exception as e:
            print(f"Warning: Error in velocity update: {e}, keeping previous velocity")
        
        try:
            # 位置更新
            p_new = self.nominal_state.p + self.nominal_state.v * dt
            
            if np.any(np.isnan(p_new)) or np.any(np.isinf(p_new)):
                print("Warning: NaN or Inf in updated position, keeping previous")
            else:
                dp = p_new - self.nominal_state.p
                dp_norm = np.linalg.norm(dp)
                if dp_norm > 2 * dt:  
                    p_new = self.nominal_state.p + dp * (2 * dt / dp_norm)
                self.nominal_state.p = p_new
        except Exception as e:
            print(f"Warning: Error in position update: {e}, keeping previous position")

    
    def propagate_error_state(self, delta_x, dt, gyro, accel):
        """Propagate error state"""
        if np.any(np.isnan(delta_x)) or np.any(np.isinf(delta_x)):
            print("Warning: NaN or Inf detected in input error state, returning zero vector")
            return np.zeros_like(delta_x)
            
        # Decompose error state vector
        delta_p = delta_x[:3]
        delta_v = delta_x[3:6]
        delta_theta = delta_x[6:9]
        delta_bg = delta_x[9:12]
        delta_ba = delta_x[12:15]
        
        theta_norm = np.linalg.norm(delta_theta)
        if theta_norm > 1.0:  
            delta_theta = delta_theta * (1.0 / theta_norm)  # Normalize to unit length
        
        bg_norm = np.linalg.norm(delta_bg)
        if bg_norm > 0.1:  # If gyro bias error is too large
            print(f"Warning: Large gyro bias error detected: {bg_norm}, clamping magnitude")
            delta_bg = delta_bg * (0.1 / bg_norm)
            
        ba_norm = np.linalg.norm(delta_ba)
        if ba_norm > 0.5: 
            print(f"Warning: Large accel bias error detected: {ba_norm}, clamping magnitude")
            delta_ba = delta_ba * (0.5 / ba_norm)
        
        # Compensate IMU bias
        gyro_corrected = gyro - (self.nominal_state.bg + delta_bg)
        accel_corrected = accel - (self.nominal_state.ba + delta_ba)
        
        # Check compensated IMU data
        if np.any(np.isnan(gyro_corrected)) or np.any(np.isinf(gyro_corrected)):
            print("Warning: NaN or Inf in corrected gyro data, using uncorrected data")
            gyro_corrected = gyro - self.nominal_state.bg
            
        if np.any(np.isnan(accel_corrected)) or np.any(np.isinf(accel_corrected)):
            print("Warning: NaN or Inf in corrected accel data, using uncorrected data")
            accel_corrected = accel - self.nominal_state.ba
        
        try:
            R_delta = so3_exp(delta_theta)
            
            # Check result
            if np.any(np.isnan(R_delta)) or np.any(np.isinf(R_delta)):
                print("Warning: NaN or Inf in R_delta, using identity")
                R_delta = np.eye(3)
                
            omega_skew = skew(gyro_corrected)
            
            # Check result
            if np.any(np.isnan(omega_skew)) or np.any(np.isinf(omega_skew)):
                print("Warning: NaN or Inf in omega_skew, using zero matrix")
                omega_skew = np.zeros((3, 3))
                
            dR_delta = np.eye(3) - dt * omega_skew
            
            # Check result
            if np.any(np.isnan(dR_delta)) or np.any(np.isinf(dR_delta)):
                print("Warning: NaN or Inf in dR_delta, using identity")
                dR_delta = np.eye(3)
                
            # Calculate new rotation error matrix
            R_delta_new = R_delta @ dR_delta
            
            # Check result
            if np.any(np.isnan(R_delta_new)) or np.any(np.isinf(R_delta_new)):
                print("Warning: NaN or Inf in R_delta_new, using identity")
                R_delta_new = np.eye(3)
                
            # Calculate new attitude error vector using so3_log
            delta_theta_new = so3_log(R_delta_new)
            
            # Check result
            if np.any(np.isnan(delta_theta_new)) or np.any(np.isinf(delta_theta_new)):
                print("Warning: NaN or Inf in delta_theta_new, using zero vector")
                delta_theta_new = np.zeros(3)
                
        except Exception as e:
            print(f"Warning: Error in attitude propagation: {e}, using simplified update")
      
            delta_theta_new = delta_theta - dt * gyro_corrected
       
            theta_norm = np.linalg.norm(delta_theta_new)
            if theta_norm > 1.0:
                delta_theta_new = delta_theta_new * (1.0 / theta_norm)
        
        try:
            R = self.nominal_state.R
            
            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                print("Warning: NaN or Inf in rotation matrix, using identity")
                R = np.eye(3)
                
            accel_nav = R.T @ accel_corrected
            
            # Check result
            if np.any(np.isnan(accel_nav)) or np.any(np.isinf(accel_nav)):
                print("Warning: NaN or Inf in accel_nav, using zero vector")
                accel_nav = np.zeros(3)
                
            accel_skew = skew(accel_nav)
            
            if np.any(np.isnan(accel_skew)) or np.any(np.isinf(accel_skew)):
                print("Warning: NaN or Inf in accel_skew, using zero matrix")
                accel_skew = np.zero
            delta_v_increment = dt * (accel_skew @ delta_theta + R.T @ delta_ba)
            
            if np.any(np.isnan(delta_v_increment)) or np.any(np.isinf(delta_v_increment)):
                print("Warning: NaN or Inf in delta_v_increment, using zero vector")
                delta_v_increment = np.zeros(3)
                
            # Update velocity error
            delta_v_new = delta_v + delta_v_increment
            
            if np.any(np.isnan(delta_v_new)) or np.any(np.isinf(delta_v_new)):
                print("Warning: NaN or Inf in delta_v_new, using previous value")
                delta_v_new = delta_v
                
        except Exception as e:
            print(f"Warning: Error in velocity propagation: {e}, keeping previous value")
            delta_v_new = delta_v
        
        try:
            # Calculate position error increment
            delta_p_increment = dt * delta_v_new
            
            # Check result
            if np.any(np.isnan(delta_p_increment)) or np.any(np.isinf(delta_p_increment)):
                print("Warning: NaN or Inf in delta_p_increment, using zero vector")
                delta_p_increment = np.zeros(3)
                
            # Update position error
            delta_p_new = delta_p + delta_p_increment
            
            if np.any(np.isnan(delta_p_new)) or np.any(np.isinf(delta_p_new)):
                print("Warning: NaN or Inf in delta_p_new, using previous value")
                delta_p_new = delta_p
                
        except Exception as e:
            # If any exception occurs, keep position error unchanged
            print(f"Warning: Error in position propagation: {e}, keeping previous value")
            delta_p_new = delta_p
        
        # Bias error propagation
        delta_bg_new = delta_bg
        delta_ba_new = delta_ba
        
        # Combine new error state
        delta_x_new = np.zeros_like(delta_x)
        delta_x_new[:3] = delta_p_new
        delta_x_new[3:6] = delta_v_new
        delta_x_new[6:9] = delta_theta_new
        delta_x_new[9:12] = delta_bg_new
        delta_x_new[12:15] = delta_ba_new
        
        # Final check of result
        if np.any(np.isnan(delta_x_new)) or np.any(np.isinf(delta_x_new)):
            print("Warning: NaN or Inf in final propagated error state, returning zero vector")
            return np.zeros_like(delta_x)
        
        return delta_x_new
    
    def update(self, z, sensor_type, i=None, return_nis=False):
        """Measurement update step
        
        Args:
            z: Measurement
            sensor_type: Sensor type ('USBL', 'DVL', 'Depth')
            i: Time step index (optional)
            return_nis: Whether to return normalized innovation squared (NIS) value
            
        Returns:
            nis: Normalized innovation squared value (if return_nis is True)
        """
        # Create copy of input data to avoid modifying original data
        z = z.copy()
        
        # 1. Local filter processing
        processed_z, R = self.local_filters[sensor_type].update(
            z, self.current_time, self.nominal_state)
        
        # Add data validity check
        if sensor_type == 'USBL':
            # USBL data anomaly detection
            if np.any(np.abs(processed_z) > 1000):  # Position jump detection
                print("USBL data anomaly, skipping update")
                return None if return_nis else None
                
        elif sensor_type == 'DVL':
            # DVL data anomaly detection
            if np.any(np.abs(processed_z) > 100):   
                print("DVL data anomaly, skipping update")
                return None if return_nis else None
        
        elif sensor_type == 'Depth':
            # Depth sensor data anomaly detection
            if np.abs(processed_z[0]) > 100:  
                print("Depth data anomaly, skipping update")
                return None if return_nis else None
            
            self.latest_depth = processed_z[0]
            
        # 2. Generate sigma points
        X = self.generate_sigma_points()
        
        # 3. Transform sigma points to measurement space
        Z = np.zeros((self.n_sigma, self.get_measurement_dim(sensor_type)))
        for i in range(self.n_sigma):
            Z[i] = self.measurement_model(X[i], sensor_type)
        
        # 4. Calculate predicted measurement mean
        z_pred = np.zeros(self.get_measurement_dim(sensor_type))
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * Z[i]
        
        # 5. Calculate measurement covariance and cross-covariance
        Pzz = np.zeros((self.get_measurement_dim(sensor_type), self.get_measurement_dim(sensor_type)))
        Pxz = np.zeros((self.n, self.get_measurement_dim(sensor_type)))
        
        for i in range(self.n_sigma):
            diff_z = Z[i] - z_pred
            diff_x = X[i] - self.delta_x
            Pzz += self.Wc[i] * np.outer(diff_z, diff_z)
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # 6. Add measurement noise
        Pzz += R
        
        Pzz = (Pzz + Pzz.T) / 2 + 1e-8 * np.eye(Pzz.shape[0])
        
        # 7. Calculate innovation
        innovation = processed_z - z_pred
        
        # Calculate normalized innovation squared (NIS)
        nis = None
        if return_nis:
            nis = innovation.T @ np.linalg.inv(Pzz) @ innovation
        
        # 8. Adaptive covariance adjustment
        if sensor_type in ['USBL', 'Depth']:  # Use adaptive adjustment for both USBL and depth sensors
            # Update sliding window
            self.fading_window.update(innovation, Pzz)
            # Calculate fading factor
            lambda_k = self.fading_window.compute_lambda()
            # Apply fading factor
            self.P = lambda_k * self.P
        
        # 9. Calculate Huber weight
        weight = huber_weight(innovation, Pzz)
        
        # 10. Calculate Kalman gain (apply Huber weight)
        K = weight * Pxz @ np.linalg.inv(Pzz)
        
        # 11. Update error state and covariance
        self.delta_x = self.delta_x + K @ innovation
        self.P = self.P - K @ Pzz @ K.T
        
        # 12. Ensure covariance matrix symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2 + 1e-8 * np.eye(self.n)
        
        # 13. Update square root covariance
        try:
            self.S = scipy.linalg.cholesky(self.P)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-10)  
            self.S = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # 14. Correct nominal state
        self.correct_nominal_state()
        
        # 15. Store navigation results
        if i is not None:
            self.nav_pos[:,i] = self.nominal_state.p.copy()
            
        # 16. Save current sensor state (for federated fusion)
        self.sensor_states[sensor_type] = {
            'x': self.delta_x.copy(),
            'P': self.P.copy(),
            'time': self.current_time
        }
        
        # 17. Try to perform federated fusion
        self.fuse_multi_sensor_data()
        
        # Return NIS value (if needed)
        if return_nis:
            return nis
    
    def correct_nominal_state(self):
        """Correct nominal state using error state"""
        # Decompose error state vector
        delta_p = self.delta_x[:3]
        delta_v = self.delta_x[3:6]
        delta_theta = self.delta_x[6:9]
        delta_bg = self.delta_x[9:12]
        delta_ba = self.delta_x[12:15]
        
        if np.any(np.isnan(self.delta_x)) or np.any(np.isinf(self.delta_x)):
            print("Warning: NaN or Inf detected in error state, skipping correction")
            self.delta_x = np.zeros(self.total_dim)
            return
        
        p_norm = np.linalg.norm(delta_p)
        if p_norm > 10.0:  
            delta_p = delta_p * (10.0 / p_norm)
        
        v_norm = np.linalg.norm(delta_v)
        if v_norm > 2.0:  
            print(f"Warning: Large velocity error detected: {v_norm}, clamping magnitude")
            delta_v = delta_v * (2.0 / v_norm)
        
        theta_norm = np.linalg.norm(delta_theta)
        if theta_norm > 0.5:  
            print(f"Warning: Large attitude error detected: {theta_norm}, clamping magnitude")
            delta_theta = delta_theta * (0.5 / theta_norm)
        
        bg_norm = np.linalg.norm(delta_bg)
        if bg_norm > 0.01: 
            print(f"Warning: Large gyro bias error detected: {bg_norm}, clamping magnitude")
            delta_bg = delta_bg * (0.01 / bg_norm)
        
        ba_norm = np.linalg.norm(delta_ba)
        if ba_norm > 0.1:  
            print(f"Warning: Large accel bias error detected: {ba_norm}, clamping magnitude")
            delta_ba = delta_ba * (0.1 / ba_norm)
        
        try:
            p_new = self.nominal_state.p + delta_p
            
            if np.any(np.isnan(p_new)) or np.any(np.isinf(p_new)):
                print("Warning: NaN or Inf in corrected position, keeping previous")
            else:
                self.nominal_state.p = p_new
            
            v_new = self.nominal_state.v + delta_v
            
            if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
                print("Warning: NaN or Inf in corrected velocity, keeping previous")
            else:
                self.nominal_state.v = v_new
            try:
                R_delta = so3_exp(delta_theta)
                
                if np.any(np.isnan(R_delta)) or np.any(np.isinf(R_delta)):
                    print("Warning: NaN or Inf in rotation increment, skipping attitude correction")
                else:
                    R_new = self.nominal_state.R @ R_delta
                    
                    if np.any(np.isnan(R_new)) or np.any(np.isinf(R_new)):
                        print("Warning: NaN or Inf in corrected rotation matrix, keeping previous")
                    else:
                        U, S, Vh = np.linalg.svd(R_new)
                        self.nominal_state.R = U @ Vh
            except Exception as e:
                print(f"Warning: Error in attitude correction: {e}, keeping previous attitude")
            
            bg_new = self.nominal_state.bg + delta_bg
            ba_new = self.nominal_state.ba + delta_ba
            
            # Check biases
            if np.any(np.isnan(bg_new)) or np.any(np.isinf(bg_new)):
                print("Warning: NaN or Inf in corrected gyro bias, keeping previous")
            else:
                self.nominal_state.bg = bg_new
                
            if np.any(np.isnan(ba_new)) or np.any(np.isinf(ba_new)):
                print("Warning: NaN or Inf in corrected accel bias, keeping previous")
            else:
                self.nominal_state.ba = ba_new
                
        except Exception as e:
            print(f"Warning: Error in state correction: {e}, keeping previous state")
        
        # Reset error state
        self.delta_x = np.zeros(self.total_dim)
    
    def measurement_model(self, delta_x, sensor_type):
        """Nonlinear measurement model
        Input:
            delta_x - Error state vector
            sensor_type - Sensor type
        Output:
            z_pred - Predicted measurement
        """
        delta_p = delta_x[:3]
        delta_v = delta_x[3:6]
        
        if sensor_type == 'USBL':
            # USBL directly measures position
            return self.nominal_state.p + delta_p
            
        elif sensor_type == 'DVL':
            # DVL directly measures velocity
            return self.nominal_state.v + delta_v
            
        elif sensor_type == 'Depth':
            return np.array([self.nominal_state.p[2] + delta_p[2]])
           
        
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
    
    def get_position(self):  
        """Get current position estimate"""
        position = self.nominal_state.p.copy()
       
        if self.latest_depth is not None and not np.isnan(self.latest_depth):
        
            z_data = np.abs(position[2] - self.latest_depth)
  
            if z_data > 0.1:
                direction = np.sign(position[2] - self.latest_depth)
                position[2] = self.latest_depth + direction * (z_data * 0.1)
        
        return position
      
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.nominal_state.v
    
    def get_euler_angles(self):
        """Get current attitude angles (Euler angles)"""
        return rotation_matrix_to_euler(self.nominal_state.R)
    
    def get_gyro_bias(self):
        """Get gyroscope bias estimate"""
        return self.nominal_state.bg
    
    def get_accel_bias(self):
        """Get accelerometer bias estimate"""
        return self.nominal_state.ba
    
    def fuse_multi_sensor_data(self, fusion_interval=5.0):
        """Federated fusion of multi-sensor data
        Input:
            fusion_interval - Fusion time interval (seconds)
        """
        # Check if fusion is needed
        if self.current_time - self.last_fusion_time < fusion_interval or len(self.sensor_states) < 2:
            return False
        
        P_list = []
        x_list = []
        sensor_types = []
        
        for sensor_type, state_info in self.sensor_states.items():
            # Check if state is expired
            if self.current_time - state_info['time'] > 2 * fusion_interval:
                continue
                
            # Extract state and covariance
            P_list.append(state_info['P'])
            x_list.append(state_info['x'])
            sensor_types.append(sensor_type)
        
        # If less than 2 valid sensors, don't perform fusion
        if len(P_list) < 2:
            return False
        
        # Calculate information allocation coefficients
        beta_list = []
        for sensor_type in sensor_types:
            if sensor_type == 'USBL':
                beta_list.append(0.4)  # Higher weight for USBL
            elif sensor_type == 'DVL':
                beta_list.append(0.4)  # Higher weight for DVL
            elif sensor_type == 'Depth':
                beta_list.append(0.2)  # Lower weight for depth sensor
            else:
                beta_list.append(1.0 / len(sensor_types))
        
        # Normalize beta values
        total_beta = sum(beta_list)
        beta_list = [beta / total_beta for beta in beta_list]
        
        # Perform federated fusion
        try:
            x_fused, P_fused = federated_fusion(P_list, x_list, beta_list)
            
            # Update global state
            self.delta_x = x_fused
            self.P = P_fused
            
            # Ensure covariance matrix symmetry and positive definiteness
            self.P = (self.P + self.P.T) / 2 + 1e-8 * np.eye(self.n)
            
            # Update square root covariance
            try:
                self.S = scipy.linalg.cholesky(self.P)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(self.P)
                eigvals = np.maximum(eigvals, 1e-10)  # Ensure all eigenvalues are positive
                self.S = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Correct nominal state
            self.correct_nominal_state()
            
            # Update fusion time
            self.last_fusion_time = self.current_time
            
            # Clear sensor state cache
            self.sensor_states = {}
            
            return True
            
        except Exception as e:
            print(f"Federated fusion failed: {e}")
            return False
    
    def reset(self):
        """Reset filter state"""
        # Reset nominal state
        self.nominal_state = NominalState()
        
        # Reset error state
        self.delta_x = np.zeros(self.total_dim)
        
        # Reset covariance matrix
        self.P = np.diag([
            5**2, 5**2, 5**2,           # Position error covariance
            0.2**2, 0.2**2, 0.5**2,      # Velocity error covariance
            (0.1)**2, (0.1)**2, (0.2)**2,  # Attitude error covariance
            (0.01)**2, (0.01)**2, (0.01)**2,  # Gyroscope bias covariance
            (0.05)**2, (0.05)**2, (0.05)**2   # Accelerometer bias covariance
        ]).astype(np.float64)
        
        # Reset square root covariance
        self.S = scipy.linalg.cholesky(self.P)
        
        # Reset adaptive filter components
        self.fading_window = FadingWindow(window_size=10)
        
        # Reset local filters
        for filter_name in self.local_filters:
            self.local_filters[filter_name].state = np.zeros(self.local_filters[filter_name].state_dim)
            self.local_filters[filter_name].P = np.eye(self.local_filters[filter_name].state_dim) * 10
            
        # Reset sensor state cache (for federated fusion)
        self.sensor_states = {}
        self.last_fusion_time = 0
            
        # Initialize sensor state cache (for federated fusion)
        self.sensor_states = {}
        self.last_fusion_time = 0
            
        # Initialize sensor state cache (for federated fusion)
        self.sensor_states = {}
        self.last_fusion_time = 0
            
        # Initialize sensor state cache (for federated fusion)
        self.sensor_states = {}
        self.last_fusion_time = 0