import numpy as np


class CoordFrameManager:
    """Coordinate transforms and sensor models for EKF tracking in NED."""

    def __init__(self, sensor_configs=None):
        self._vessel_pos = np.array([0.0, 0.0], dtype=float)

        sensor_configs = sensor_configs or {}

        radar_cfg = sensor_configs.get("radar", {})
        cam_cfg = sensor_configs.get("camera", {})

        self.radar_pos = np.array(radar_cfg.get("pos_ned", [0.0, 0.0]), dtype=float)
        self.camera_pos = np.array(cam_cfg.get("pos_ned", [-80.0, 120.0]), dtype=float)

        self.sigma_r_radar = float(radar_cfg.get("sigma_r_m", 5.0))
        self.sigma_phi_radar = np.deg2rad(float(radar_cfg.get("sigma_phi_deg", 0.3)))

        self.sigma_r_camera = float(cam_cfg.get("sigma_r_m", 8.0))
        self.sigma_phi_camera = np.deg2rad(float(cam_cfg.get("sigma_phi_deg", 0.15)))

        self.camera_boresight = np.deg2rad(float(cam_cfg.get("boresight_deg", 0.0)))
        self.camera_fov = np.deg2rad(float(cam_cfg.get("fov_deg", 180.0)))
        self.camera_max_range = float(cam_cfg.get("range_m", 500.0))

        self.radar_fov = np.deg2rad(float(radar_cfg.get("fov_deg", 360.0)))
        self.radar_max_range = float(radar_cfg.get("range_m", np.inf))

        ais_cfg = sensor_configs.get("ais", {})
        self.sigma_pos_ais = float(ais_cfg.get("sigma_pos_m", 4.0))

    def update_vessel_pos(self, north_m: float, east_m: float) -> None:
        self._vessel_pos = np.array([float(north_m), float(east_m)], dtype=float)

    @property
    def vessel_pos(self) -> np.ndarray:
        return self._vessel_pos.copy()

    def h(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        s = self._sensor_pos(sensor_id)
        return self._range_bearing(x, s)

    def H(self, x: np.ndarray, sensor_id: str) -> np.ndarray:
        s = self._sensor_pos(sensor_id)
        dN = x[0] - s[0]
        dE = x[1] - s[1]
        r = np.sqrt(dN**2 + dE**2)

        if r < 1e-6:
            return np.zeros((2, 4), dtype=float)

        dr_dpN = dN / r
        dr_dpE = dE / r
        dphi_dpN = -dE / (r**2)
        dphi_dpE = dN / (r**2)

        return np.array(
            [
                [dr_dpN, dr_dpE, 0.0, 0.0],
                [dphi_dpN, dphi_dpE, 0.0, 0.0],
            ],
            dtype=float,
        )

    def R(self, sensor_id: str) -> np.ndarray:
        if sensor_id == "radar":
            return np.diag([self.sigma_r_radar**2, self.sigma_phi_radar**2])
        if sensor_id == "camera":
            return np.diag([self.sigma_r_camera**2, self.sigma_phi_camera**2])
        if sensor_id == "ais":
            return np.diag([self.sigma_pos_ais**2, self.sigma_pos_ais**2])
        raise ValueError(f"Unknown sensor_id '{sensor_id}'.")

    def measurement_in_fov_and_range(self, sensor_id: str, z: np.ndarray) -> bool:
        """Validate raw polar measurement against sensor FOV/range limits."""
        z = np.asarray(z, dtype=float).reshape(-1)
        r = float(z[0])
        phi = float(z[1])

        if sensor_id == "camera":
            if r < 0.0 or r > self.camera_max_range:
                return False
            half_fov = 0.5 * self.camera_fov
            rel = self._wrap_angle(phi - self.camera_boresight)
            return abs(rel) <= half_fov

        if sensor_id == "radar":
            if r < 0.0 or r > self.radar_max_range:
                return False
            if self.radar_fov >= 2 * np.pi - 1e-9:
                return True
            half_fov = 0.5 * self.radar_fov
            return abs(self._wrap_angle(phi)) <= half_fov

        return True

    def ais_ned_to_range_bearing(self, north_m: float, east_m: float) -> np.ndarray:
        target_pos = np.array([north_m, east_m], dtype=float)
        dN = target_pos[0] - self._vessel_pos[0]
        dE = target_pos[1] - self._vessel_pos[1]
        r = np.sqrt(dN**2 + dE**2)
        phi = np.arctan2(dE, dN)
        return np.array([r, phi], dtype=float)

    def _sensor_pos(self, sensor_id: str) -> np.ndarray:
        if sensor_id == "radar":
            return self.radar_pos
        if sensor_id == "camera":
            return self.camera_pos
        if sensor_id == "ais":
            return self._vessel_pos
        raise ValueError(f"Unknown sensor_id '{sensor_id}'.")

    @staticmethod
    def _range_bearing(x: np.ndarray, s: np.ndarray) -> np.ndarray:
        dN = x[0] - s[0]
        dE = x[1] - s[1]
        r = np.sqrt(dN**2 + dE**2)
        phi = np.arctan2(dE, dN)
        return np.array([r, phi], dtype=float)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)
