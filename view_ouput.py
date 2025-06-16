import matplotlib.pyplot as plt
import numpy as np


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = R_z @ R_y @ R_x
    return R


# Wczytaj dane
poses = np.load("rgb_trajectory.npy")  # zakładam shape (N, 3) lub (N, 6)
gt = np.loadtxt("falcon_indoor_flight_1_pose_evo_gt.txt")

# Ground truth
gt_x = gt[:, 1]
gt_y = gt[:, 3]
gt_z = gt[:, 2]

# Jeśli poses ma shape (N, 3) - tylko translacja
if poses.shape[1] == 3:
    abs_poses = [poses[0]]
    for i in range(1, len(poses)):
        abs_poses.append(abs_poses[-1] + poses[i])
    abs_poses = np.array(abs_poses)
    x = abs_poses[:, 0]
    y = abs_poses[:, 1]
    z = abs_poses[:, 2]
# Jeśli poses ma shape (N, 6) - [yaw, pitch, roll, x, y, z]
elif poses.shape[1] == 6:
    abs_pose = np.zeros(6)
    abs_poses = []
    for rel_pose in poses:
        # Translacja w układzie lokalnym
        R = eulerAnglesToRotationMatrix(abs_pose[:3])
        trans = R @ rel_pose[3:]
        abs_pose[:3] += rel_pose[:3]  # sumuj kąty
        abs_pose[3:] += trans  # sumuj przesunięcia
        abs_poses.append(abs_pose.copy())
    abs_poses = np.array(abs_poses)
    x = abs_poses[:, 3]
    y = abs_poses[:, 4]
    z = abs_poses[:, 5]
else:
    raise ValueError("Nieznany format poses")

# Rysowanie
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")



ax.plot(x, y, z, linewidth=2, label="Estimated")
ax.plot(gt_x, gt_y, gt_z, color="red", linewidth=2, label="Ground Truth")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Estimated Trajectory (3D)")
ax.legend()
plt.show()
