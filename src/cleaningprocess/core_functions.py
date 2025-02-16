
import numpy as np

class BallData:
    def __init__(self, time, positions, velocities, accelerations):
        self.time = time
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.accelerations = np.array(accelerations)

    def calculate_curvature(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        cross = np.cross(v1, v2)
        return 2 * np.linalg.norm(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v1 + v2))

    def calculate_metrics(self):
        # Overall metrics
        total_time = self.time[-1] - self.time[0]
        displacement = self.positions[-1] - self.positions[0]
        total_distance = np.sum(np.sqrt(np.sum(np.diff(self.positions, axis=0)**2, axis=1)))
        average_speed = total_distance / total_time
        average_velocity = displacement / total_time

        max_speed = np.max(np.sqrt(np.sum(self.velocities**2, axis=1)))
        min_speed = np.min(np.sqrt(np.sum(self.velocities**2, axis=1)))

        max_acceleration = np.max(np.sqrt(np.sum(self.accelerations**2, axis=1)))
        min_acceleration = np.min(np.sqrt(np.sum(self.accelerations**2, axis=1)))

        # Trajectory context metrics
        instantaneous_speeds = np.sqrt(np.sum(self.velocities**2, axis=1))

        # Peak height (assuming Z is the vertical axis)
        peak_height = np.max(self.positions[:, 2])
        time_to_peak = self.time[np.argmax(self.positions[:, 2])] - self.time[0]

        # Range (assuming X is the primary horizontal axis)
        range_distance = np.abs(self.positions[-1, 0] - self.positions[0, 0])

        # Angle of elevation (in radians)
        elevation_angles = np.arctan2(self.velocities[:, 2], np.sqrt(self.velocities[:, 0]**2 + self.velocities[:, 1]**2))

        # Curvatures (using the calculate_curvature method)
        curvatures = [self.calculate_curvature(self.positions[i], self.positions[i+1], self.positions[i+2])
                      for i in range(len(self.positions)-2)]

        # Construct the metrics dictionary
        metrics_dict = {
            'total_time': total_time,
            'total_distance': total_distance,
            'average_speed': np.linalg.norm(average_speed),
            'max_speed': max_speed,
            'peak_height': peak_height,
            'time_to_peak_height': time_to_peak,
            'range': range_distance,
            'positions': self.positions,
            'velocities': self.velocities,
            'accelerations': self.accelerations,
            'times': self.time
        }

        return metrics_dict
