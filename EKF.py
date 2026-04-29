"""
T3 — Single-Target EKF
======================
Minimal constant-velocity EKF for tracking a single target in NED coordinates.

State: x = [p_N, p_E, v_N, v_E]
Measurement: z = [range, bearing] from radar

The EKF delegates all sensor geometry (h, H, R) to CoordFrameManager.
"""
import numpy as np
from coord_frame_manager import CoordFrameManager


class EKF:
	"""Single-target constant-velocity EKF."""

	def __init__(self, cfm=None, sigma_a=0.05):
		"""
		Initialize an EKF.
		
		Parameters:
		  cfm       : CoordFrameManager instance (or None → create new)
		  sigma_a   : process noise std dev [m/s²]
		"""
		self.cfm = cfm if cfm is not None else CoordFrameManager()
		self.sigma_a = float(sigma_a)
		
		# State and covariance
		self.x = np.zeros(4, dtype=float)  # [p_N, p_E, v_N, v_E]
		self.P = np.eye(4, dtype=float)

	def predict(self, dt):
		"""
		Propagate state and covariance forward by dt seconds.
		
		Uses constant-velocity model:
		  F = [[1, 0, dt, 0],
		       [0, 1, 0, dt],
		       [0, 0, 1, 0],
		       [0, 0, 0, 1]]
		  
		  Q = process noise covariance (DWNA model)
		"""
		dt = float(dt)
		
		# State transition matrix
		F = np.array([
			[1.0, 0.0, dt,  0.0],
			[0.0, 1.0, 0.0, dt],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		], dtype=float)
		
		# Process noise covariance (DWNA: Discrete White Noise Acceleration)
		dt2, dt3, dt4 = dt**2, dt**3, dt**4
		q = self.sigma_a**2
		Q = q * np.array([
			[dt4/4, 0,     dt3/2, 0],
			[0,     dt4/4, 0,     dt3/2],
			[dt3/2, 0,     dt2,   0],
			[0,     dt3/2, 0,     dt2],
		], dtype=float)
		
		# Propagate
		self.x = F @ self.x
		self.P = F @ self.P @ F.T + Q
		self.P = 0.5 * (self.P + self.P.T)  # Symmetrize

	def update(self, z, sensor_id):
		"""
		Apply a measurement update.
		
		Parameters:
		  z          : measurement vector [range, bearing]
		  sensor_id  : 'radar'
		  
		Returns:
		  nis        : Normalized Innovation Squared for validation
		"""
		z = np.asarray(z, dtype=float).reshape(-1)
		
		# Get measurement model and noise covariance from coordinate manager
		h_pred = self.cfm.h(self.x, sensor_id)
		H = self.cfm.H(self.x, sensor_id)
		R = self.cfm.R(sensor_id)
		
		# Innovation (measurement residual)
		y = z - h_pred
		
		# Wrap bearing component if present
		if y.size >= 2:
			y[1] = float((y[1] + np.pi) % (2 * np.pi) - np.pi)
		
		# Innovation covariance
		S = H @ self.P @ H.T + R
		
		# Kalman gain
		K = self.P @ H.T @ np.linalg.inv(S)
		
		# Update state and covariance
		self.x = self.x + K @ y
		I = np.eye(4, dtype=float)
		self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
		self.P = 0.5 * (self.P + self.P.T)  # Symmetrize
		
		# Normalized Innovation Squared (NIS) for validation
		nis = float(y.T @ np.linalg.inv(S) @ y)
		return nis

	def initialize_from_measurement(self, z, sensor_id):
		"""
		Set initial state and covariance from first measurement.
		
		For radar: converts [range, bearing] to NED position.
		Velocity is initialized to zero.
		"""
		if sensor_id != 'radar':
			raise ValueError(f"Task 3: radar only. Got '{sensor_id}'")
		
		z = np.asarray(z, dtype=float).reshape(-1)
		
		# Extract sensor position and convert to NED
		s = self.cfm._sensor_pos(sensor_id)
		r, phi = float(z[0]), float(z[1])
		p_n = s[0] + r * np.cos(phi)
		p_e = s[1] + r * np.sin(phi)
		
		self.x = np.array([p_n, p_e, 0.0, 0.0], dtype=float)
		
		# Covariance: propagate measurement uncertainty to position space
		# Then add high velocity uncertainty (unknown from single measurement)
		R = self.cfm.R(sensor_id)
		J = np.array([
			[np.cos(phi), -r * np.sin(phi)],
			[np.sin(phi),  r * np.cos(phi)],
		], dtype=float)
		pos_cov = J @ R @ J.T
		
		self.P = np.diag([
			pos_cov[0, 0],  # p_N variance
			pos_cov[1, 1],  # p_E variance
			100.0,          # v_N variance (large, unknown)
			100.0,          # v_E variance (large, unknown)
		])

	def state(self):
		"""Return current state estimate [p_N, p_E, v_N, v_E]."""
		return self.x.copy()

	def covariance(self):
		"""Return current covariance matrix."""
		return self.P.copy()
