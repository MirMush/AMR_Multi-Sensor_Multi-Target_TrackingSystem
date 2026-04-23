"""
T2 — Coordinate Frame Manager
==============================
  - Radar   → (range, bearing) measured from the radar position [0, 0]
  - Camera  → (range, bearing) measured from the camera position [-80, 120]
  - AIS     → (north, east) absolute NED position of the target
  - GNSS    → (north, east) absolute NED position of OUR OWN vessel

The EKF tracks targets in a single shared NED frame with state:
    x = [p_N, p_E, v_N, v_E]

For each sensor,it provides:
    h(x)  — what would this sensor predict to measure, given state x?
    H     — the Jacobian (how sensitive is h to changes in x?)
    R     — the measurement noise covariance matrix

The EKF needs all three of these every time it processes a measurement.
"""
import numpy as np
# =============================================================================
# Sensor positions in the NED frame (metres)
# Origin of the NED frame is defined as the radar position.
# These are fixed constants from the project specification.
# =============================================================================
# Radar sits at the NED origin, so its offset is zero in both axes
RADAR_POS = np.array([0.0, 0.0])
# Camera is mounted 80 m south and 120 m east of the radar
CAMERA_POS = np.array([-80.0, 120.0])
# =============================================================================
# Sensor noise standard deviations 
# These describe how noisy each sensor's measurements are.
# =============================================================================
# Radar: 5 m range noise, 0.3 degrees bearing noise (converted to radians)
SIGMA_R_RADAR   = 5.0
SIGMA_PHI_RADAR = 0.3 * (np.pi / 180.0)   # deg to rad conversion
# Camera: 8 m range noise, 0.15 degrees bearing noise (finer than radar)
SIGMA_R_CAMERA   = 8.0
SIGMA_PHI_CAMERA = 0.15 * (np.pi / 180.0) # deg to rad conversion
# AIS: 4 m position noise in NED (both north and east axes)
SIGMA_POS_AIS = 4.0
# GNSS: 2 m position noise — used for the vessel's own position
SIGMA_POS_GNSS = 2.0
# =============================================================================
# Main class
# =============================================================================
class CoordFrameManager:
    """
    Computes h(x), H, and R for each sensor so the EKF can process
    measurements from all four sensors in a unified way.
    """
    def __init__(self):
        # Initialise the vessel position estimate to the NED origin.
        # This will be updated every time a GNSS fix arrives.
        self._vessel_pos = np.array([0.0, 0.0])
    # -------------------------------------------------------------------------
    # GNSS update — call this every time a new GNSS fix arrives
    # -------------------------------------------------------------------------
    def update_vessel_pos(self, north_m: float, east_m: float) -> None: #it reaches inside the class and changes the value of a "memory box" so the other functions (h and H) can use the correct numbers later.
        """
        Store the latest GNSS fix as the current vessel position.
        This is needed by the AIS model because AIS measurements are
        expressed relative to wherever the vessel currently is.
        """
        # Overwrite the stored vessel position with the new GNSS fix
        self._vessel_pos = np.array([float(north_m), float(east_m)])
    @property #this decorator turns the method in a read-only attribute.
    def vessel_pos(self) -> np.ndarray: #ensures that When you call this property, you are guaranteed to receive a NumPy array
        # Read-only access to the current vessel position estimate
        #every time cfm.vessel_pos is typed, Python automatically runs this line of code and gives you the current value of the secret storage.
        return self._vessel_pos.copy() #returns a copy to avoid risking to modify original data 
    # -------------------------------------------------------------------------
    # h(x, sensor_id) — predicted measurement given target state x
    # -------------------------------------------------------------------------
    def h(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        """
        Returns the predicted measurement vector [range_m, bearing_rad]
        for the given sensor, evaluated at state x = [p_N, p_E, v_N, v_E].
        This answers the question: "If the target is at state x, what
        would sensor sensor_id expect to measure?"
        """
        # Look up where this sensor is located in the NED frame
        s = self._sensor_pos(sensor_id)
        # Compute and return the predicted (range, bearing) using the
        # geometry between the target position and the sensor position
        return self._range_bearing(x, s) #ensures actual sight and prediction are consistent. 
        #x is the state vector [p_N, p_E, v_N, v_E]
        #s is the sensor position in NED [s_N, s_E]
    # -------------------------------------------------------------------------
    # H(x, sensor_id) — Jacobian of h with respect to x
    # -------------------------------------------------------------------------
    def H(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        """
        Returns the 2x4 Jacobian matrix dh/dx evaluated at state x.

        The EKF linearises the nonlinear measurement function h around
        the current state estimate. This linearisation IS the Jacobian.
        It tells the filter how much each element of the measurement
        changes when each element of the state changes slightly.

        Layout:
            Row 0: d(range)   / d[p_N, p_E, v_N, v_E]
            Row 1: d(bearing) / d[p_N, p_E, v_N, v_E]
        """
        # Get sensor position to compute the offset to the target
        s = self._sensor_pos(sensor_id)
        # Delta north and delta east: vector from sensor to target
        dN = x[0] - s[0]   # target north minus sensor north, x[0] and s[0] are the north components of the target and sensor positions given 
        #x = [p_N, p_E, v_N, v_E] where x[0] = p_N (target north position),x[1] = p_E (target east position) ecc..
        #s = [s_N, s_E]
        #     [0]   [1]

        dE = x[1] - s[1]   # target east  minus sensor east
        # Euclidean distance from sensor to target (the true range)
        r = np.sqrt(dN**2 + dE**2) #dN is y axis and dE is x axis, so this is the Pythagorean theorem to get the straight-line distance from sensor to target.
        # Guard against division by zero if target is on top of sensor
        if r < 1e-6:
            # Jacobian is undefined here; return a zero matrix as a safe fallback
            return np.zeros((2, 4))
        # --- Partial derivatives of RANGE r = sqrt(dN^2 + dE^2) ---
        # d(r)/d(p_N) = dN / r
        dr_dpN = dN / r #partial derivative representing the rate of change of the distance (range r) relative to North direction
        # d(r)/d(p_E) = dE / r
        dr_dpE = dE / r #partial derivative representing the rate of change of the distance (range r )relative to East direction
        # Velocity components v_N, v_E partial derivatives are zero
        
        # --- Partial derivatives of BEARING phi = atan2(dE, dN) ---
        # Using standard atan2 derivative rules:
        #   d(atan2(dE,dN))/d(dN) = -dE / r^2
        #   d(atan2(dE,dN))/d(dE) =  dN / r^2
        # Since dN = p_N - s_N and d(dN)/d(p_N) = 1:
        dphi_dpN = -dE / (r**2)   # bearing becomes more negative as we go north
        dphi_dpE =  dN / (r**2)   # bearing increases as we go east

        # The four partial derivatives are stacked to formthe 2x4 Jacobian matrix.
        # Columns 0,1 are position partials; columns 2,3 are velocity partials (zero).
        Hmat = np.array([
            [dr_dpN,   dr_dpE,   0.0, 0.0],  # row 0: how range changes with state
            [dphi_dpN, dphi_dpE, 0.0, 0.0],  # row 1: how bearing changes with state
        ])
        return Hmat

    # -------------------------------------------------------------------------
    # R(sensor_id) — measurement noise covariance matrix
    # -------------------------------------------------------------------------

    def R(self, sensor_id: str) -> np.ndarray:
        """
        Returns the 2x2 diagonal noise covariance matrix for the sensor.

        R tells the EKF how much to trust this sensor's measurements.
        Large diagonal values = noisy sensor = less trust.
        Small diagonal values = precise sensor = more trust.

        Units: [m^2, rad^2] — variances, so the std deviations are squared.
        """
        if sensor_id == 'radar':
            # Radar: range variance and bearing variance on the diagonal.
            # Off-diagonal is zero because range and bearing errors are independent.
            return np.diag([
                SIGMA_R_RADAR**2,     # range noise variance [m^2]
                SIGMA_PHI_RADAR**2,   # bearing noise variance [rad^2]
            ])

        elif sensor_id == 'camera':
            return np.diag([
                SIGMA_R_CAMERA**2,    # range noise variance [m^2]
                SIGMA_PHI_CAMERA**2,  # bearing noise variance [rad^2]
            ])

        elif sensor_id == 'ais':
            # AIS noise is specified as sigma_pos = 4 m in NED position space.
            # AIS is converted to (range, bearing):
            #   range noise   ~ sigma_pos directly (radial direction)
            #   bearing noise ~ sigma_pos / r (arc length divided by radius)
            return np.diag([
                SIGMA_POS_AIS**2,   # range noise variance [m^2]
                SIGMA_POS_AIS**2,   # bearing noise variance (conservative) [m^2]
            ])

        else:
            raise ValueError(
                f"Unknown sensor_id '{sensor_id}'. "
                f"Valid options are: 'radar', 'camera', 'ais'."
            )

    # -------------------------------------------------------------------------
    # AIS helper — converts a raw AIS NED report into (range, bearing)
    # -------------------------------------------------------------------------

    def ais_ned_to_range_bearing(self,
                                  north_m: float,
                                  east_m: float) -> np.ndarray:
        """
        AIS reports give an absolute NED position of the target [north, east].
        But measurement model h(x, 'ais') outputs (range, bearing)
        relative to the vessel, so the raw AIS report is converted
        into the same format before computing the innovation y - h(x).
        This function does that conversion using the current vessel position.
        """
        # Build the target position vector from the AIS report
        target_pos = np.array([north_m, east_m])

        # Compute the vector from the vessel to the target in NED components
        dN = target_pos[0] - self._vessel_pos[0]  # how far north the target is
        dE = target_pos[1] - self._vessel_pos[1]  # how far east  the target is

        # Convert to polar coordinates: straight-line distance
        r   = np.sqrt(dN**2 + dE**2)

        # Bearing from North, clockwise — matches the NED bearing convention
        phi = np.arctan2(dE, dN)

        # Return as a 2-element measurement vector matching h(x, 'ais') format
        return np.array([r, phi])
    # -------------------------------------------------------------------------
    #         helper functions
    # -------------------------------------------------------------------------
    def _sensor_pos(self, sensor_id: str) -> np.ndarray:
        """
        Return the NED position [north, east] of the requested sensor.
        For AIS the position is the vessel's current GNSS position because
        the AIS receiver is mounted on the vessel and moves with it.
        """
        if sensor_id == 'radar':
            return RADAR_POS              # fixed at NED origin [0, 0]
        elif sensor_id == 'camera':
            return CAMERA_POS             # fixed land-based offset [-80, 120]
        elif sensor_id == 'ais':
            return self._vessel_pos       # moves with the vessel every GNSS fix
        else:
            raise ValueError(f"Unknown sensor_id '{sensor_id}'.")

    @staticmethod #this decorator indicates that the method does not depend on the instance (self) and can be called on the class itself without needing an instance.
    def _range_bearing(x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Core geometry function used by h().
        Computes [range, bearing] from sensor position s to target state x.

        x = [p_N, p_E, v_N, v_E]  — target state (only position is used here)
        s = [s_N, s_E]             — sensor position in NED
        """
        # Separation vector from sensor to target, split into NED components
        dN = x[0] - s[0]   # north separation between target and sensor
        dE = x[1] - s[1]   # east  separation between target and sensor

        # Straight-line distance (range) from sensor to target
        r = np.sqrt(dN**2 + dE**2)

        # Bearing angle from North, measured clockwise.
        # atan2(east_component, north_component) gives angle from North axis.
        phi = np.arctan2(dE, dN)

        # Return as a 2-element vector [range_m, bearing_rad]
        return np.array([r, phi])
# =============================================================================
# Unit Tests
# =============================================================================

def _run_unit_tests():
    print("=" * 60)
    print("T2 Unit Tests — CoordFrameManager")
    print("=" * 60)

    cfm    = CoordFrameManager()
    passed = 0
    failed = 0

    def check(name, got, expected, tol=1e-6):
        """Helper: compare arrays, print PASS or FAIL with error magnitude."""
        nonlocal passed, failed
        err = np.max(np.abs(np.array(got) - np.array(expected)))
        if err < tol:
            print(f"  PASS  {name}  (max err={err:.2e})")
            passed += 1
        else:
            print(f"  FAIL  {name}  (max err={err:.2e})")
            print(f"        expected {expected}")
            print(f"        got      {got}")
            failed += 1

    # ------------------------------------------------------------------
    # Test 1 — Radar h(x)
    # Target at [300, 400] m NED, radar at [0, 0]
    # Expected: range = sqrt(300^2 + 400^2) = 500 m
    #           bearing = atan2(400, 300)
    # ------------------------------------------------------------------
    x = np.array([300.0, 400.0, 0.0, 0.0])
    y = cfm.h(x, 'radar')
    check("T1a  radar range   (expect 500 m)",          y[0], 500.0)
    check("T1b  radar bearing (expect atan2(400,300))", y[1], np.arctan2(400, 300))

    # ------------------------------------------------------------------
    # Test 2 — Camera h(x)
    # Same target [300, 400], camera at [-80, 120]
    # dN = 300-(-80) = 380,  dE = 400-120 = 280
    # ------------------------------------------------------------------
    dN_c  = 300.0 - (-80.0)   # = 380 m north separation
    dE_c  = 400.0 - 120.0     # = 280 m east  separation
    y_cam = cfm.h(x, 'camera')
    check("T2a  camera range   (expect sqrt(380^2+280^2))",
          y_cam[0], np.sqrt(dN_c**2 + dE_c**2))
    check("T2b  camera bearing (expect atan2(280,380))",
          y_cam[1], np.arctan2(dE_c, dN_c))

    # ------------------------------------------------------------------
    # Test 3 — AIS consistency with radar
    # When vessel is at [0,0], AIS sensor pos == radar pos,
    # so h(x,'ais') must equal h(x,'radar') exactly.
    # ------------------------------------------------------------------
    cfm.update_vessel_pos(0.0, 0.0)   # vessel at NED origin, same as radar
    y_ais   = cfm.h(x, 'ais')
    y_radar = cfm.h(x, 'radar')
    check("T3a  AIS range   == radar range   (vessel at origin)", y_ais[0], y_radar[0])
    check("T3b  AIS bearing == radar bearing (vessel at origin)", y_ais[1], y_radar[1])

    # Test the AIS NED to range-bearing conversion helper with vessel offset
    cfm.update_vessel_pos(100.0, 50.0)   # move vessel to [100, 50]
    y_conv = cfm.ais_ned_to_range_bearing(300.0, 400.0)
    # dN = 300-100 = 200,  dE = 400-50 = 350
    check("T3c  ais_ned_to_rb range",   y_conv[0], np.sqrt(200**2 + 350**2))
    check("T3d  ais_ned_to_rb bearing", y_conv[1], np.arctan2(350, 200))

    # ------------------------------------------------------------------
    # Test 4 — Jacobian H verified by finite differences
    # Numerically approximate dh/dx and compare to analytical H.
    # Agreement to ~1e-8 confirms the hand-derived formula is correct.
    # ------------------------------------------------------------------
    cfm.update_vessel_pos(0.0, 0.0)
    x0  = np.array([300.0, 400.0, 1.5, -2.0])
    eps = 1e-4   # small perturbation step for finite difference

    for sid in ['radar', 'camera']:
        H_ana = cfm.H(x0, sid)    # analytical Jacobian
        H_num = np.zeros((2, 4))  # numerical Jacobian to compare against

        for col in range(4):
            
            xp = x0.copy(); xp[col] += eps   # state + epsilon
            xm = x0.copy(); xm[col] -= eps   # state - epsilon
            # Central difference: (h(x+eps) - h(x-eps)) / (2*eps)
            H_num[:, col] = (cfm.h(xp, sid) - cfm.h(xm, sid)) / (2 * eps)

        check(f"T4   {sid} Jacobian matches finite differences", H_ana, H_num, tol=1e-5)

    # ------------------------------------------------------------------
    # Test 5 — R matrices: correct shape, diagonal, positive definite
    # ------------------------------------------------------------------
    for sid in ['radar', 'camera', 'ais']:
        R = cfm.R(sid)
        # Shape must be 2x2 to match the 2-element measurement vector
        assert R.shape == (2, 2),       f"R shape wrong for {sid}"
        # Diagonal entries must be strictly positive (variances > 0)
        assert np.all(np.diag(R) > 0), f"R not positive definite for {sid}"
        # Off-diagonal must be zero (range and bearing noise are independent)
        assert R[0,1] == 0 and R[1,0] == 0, f"R not diagonal for {sid}"
        print(f"  PASS  T5   {sid} R is 2x2, diagonal, positive-definite")
        passed += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    _run_unit_tests()
