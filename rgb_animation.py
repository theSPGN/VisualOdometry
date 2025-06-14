import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


FILE_PATH = "falcon_indoor_flight_1_data.h5"


if __name__ == "__main__":

    show_ref = False

    file_trajectory = "rgb_trajectory.npy"
    image_format = "rgb"

    # file_trajectory = "event_trajectory.npy"
    # image_format = "events"


    trajectory_3d_np = np.load(file_trajectory)
    if show_ref:
        ground_truth = np.loadtxt("falcon_indoor_flight_1_pose_evo_gt.txt")


    file = h5py.File(FILE_PATH, "r") 
    rgb_dataset = file["ovc"][image_format]["data"]
    num_images = rgb_dataset.shape[0]


    rgb = np.array(rgb_dataset[0], dtype=np.uint8)
    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


    x_data = trajectory_3d_np[:, 0]
    y_data = trajectory_3d_np[:, 1]
    h_data = trajectory_3d_np[:, 2]
    t_data = np.arange(0, num_images)

    if show_ref:
        ref_x_data = ground_truth[:, 1]
        ref_y_data = ground_truth[:, 2]
        ref_h_data = ground_truth[:, 3]
        ref_t_data = np.arange(0, len(ref_h_data))

    print(f"img: {num_images}, len_trajectory: {trajectory_3d_np.shape}")


    fig = plt.figure()
    fig.set_size_inches(16, 10)
    fig.tight_layout()
    plt.gray()

    gs = GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[3, 2])

    img_ax = fig.add_subplot(gs[0, :])
    xy_ax = fig.add_subplot(gs[1, 0])
    h_ax = fig.add_subplot(gs[1, 1])

    img_plot = img_ax.imshow(gray0)

    xy_plot, = xy_ax.plot([], [])
    ref_xy_plot, = xy_ax.plot([], [], 'r--')

    h_plot, = h_ax.plot([], [])
    ref_h_plot, = h_ax.plot([], [], 'r--')

    xlim = [min(x_data), max(x_data)]
    ylim = [min(y_data), max(y_data)]
    hlim = [min(h_data), max(h_data)]

    if show_ref:
        xlim = [min([xlim[0], min(ref_x_data)]), max([xlim[1], max(ref_x_data)])]
        ylim = [min([ylim[0], min(ref_y_data)]), max([ylim[1], max(ref_y_data)])]
        hlim = [min([hlim[0], min(ref_h_data)]), max([hlim[1], max(ref_h_data)])]


    xy_ax.set(xlim=xlim, ylim=ylim, xlabel='X [m]', ylabel='Y [m]')
    h_ax.set(xlim=[0, num_images], ylim=hlim, xlabel='frame', ylabel='H [m]')

    def update(frame):
        rgb = np.array(rgb_dataset[frame], dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        img_plot.set_data(gray)
        xy_plot.set_data(x_data[:frame], y_data[:frame])
        h_plot.set_data(t_data[:frame], h_data[:frame])

        if show_ref:
            ref_xy_plot.set_data(ref_x_data, ref_y_data)
            ref_h_plot.set_data(ref_t_data, ref_h_data)

        return img_plot, xy_plot, h_plot, ref_xy_plot, ref_h_plot

    ani = FuncAnimation(fig, update, frames=num_images, interval=10, blit=True)
    plt.show()
    file.close()
