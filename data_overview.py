# %%
import h5py
import numpy as np
import cv2

"""
ouster:
['calib', 'data', 'imu', 'metadata', 'ts_end', 'ts_start', 'ts_start_map_prophesee_left_t', 'ts_start_map_prophesee_right_t']
ovc:
['imu', 'left', 'rgb', 'right', 'ts', 'ts_map_prophesee_left_t', 'ts_map_prophesee_right_t']
prophesee:
['left', 'right']
"""

FILE_PATH = "falcon_indoor_flight_1_data.h5"

with h5py.File(FILE_PATH, "r") as file:
    # for item in (list(file.keys())):
    #     print(item,"items:")
    #     print(list(file[item]))
    data = np.array(list(file["ouster"]["data"]))
    ovc_rgb = np.array(list(file["ovc"]["rgb"]["data"]))
    ts = np.array(list(file["ovc"]["ts"]))
    left_dvs = [np.array(list(file["prophesee"]["left"]['calib'])),
                np.array(list(file["prophesee"]["left"]['p'])),
                np.array(list(file["prophesee"]["left"]['t'])),
                np.array(list(file["prophesee"]["left"]['x'])),
                np.array(list(file["prophesee"]["left"]['y']))]
    right_dvs = np.array(list(file["prophesee"]["right"]))

# %%
with h5py.File(FILE_PATH, "r") as file:
    left_cam = np.array(list(file["ovc"]["left"]["data"]))
    right_cam = np.array(list(file["ovc"]["right"]["data"]))

# %%
with h5py.File(FILE_PATH, "r") as file:
    calib = np.array(list(file["ouster"]["calib"]["T_to_prophesee_left"]))
    imu_ovc = [
        np.array(list(file["ouster"]["imu"]["accel"])),
        np.array(list(file["ouster"]["imu"]["omega"])),
        np.array(list(file["ouster"]["imu"]["ts"])),
    ]
# %%
data = np.reshape(data, (12609, 570, 128))
# %%
print(data.shape, ovc_rgb.shape, ts.shape)
sampling_rate = (ts[1] - ts[0]) / 1000
# %%
for i in range(12609):
    img = data[i, :, :]
    print(i)
    cv2.imshow("zdj", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()

# %%
for i in range(1424):
    img = ovc_rgb[i, :, :, :]
    cv2.imshow("zdj", img)
    # left = left_cam[i,:,:,:]
    # right= right_cam[i,:,:,:]
    # cv2.imshow("left", left)
    # cv2.imshow("right", right)
    cv2.waitKey(int(sampling_rate))
cv2.destroyAllWindows()
