import numpy as np
import h5py
import cv2
import open3d as o3d

FILE_PATH = r"C:\VisualOdometry\falcon_indoor_flight_1_data.h5"

# Konfiguracja StereoBM – dysparycja musi być wielokrotnością 16
numDisparities = 64  # przykładowa wartość; można eksperymentalnie dobrać np. 16, 32, 64
blockSize = 15  # zwykle nieparzysta liczba
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

with h5py.File(FILE_PATH, "r") as file:
    # Pobieramy parametry kalibracji dla RGB
    fx, fy, cx, cy = list(file["ovc"]["rgb"]["calib"]["intrinsics"])

    # Wczytujemy zestawy obrazów
    rgb_dataset = file["ovc"]["rgb"]["data"]
    left_dataset = file["ovc"]["left"]["data"]
    right_dataset = file["ovc"]["right"]["data"]
    num_images = rgb_dataset.shape[0]

    # Ustal rozdzielczość obrazu RGB (zakładamy, że kształt obrazu to [wysokość, szerokość, 3])
    rgb_sample = rgb_dataset[0]
    rgb_h, rgb_w = rgb_sample.shape[0], rgb_sample.shape[1]

    # Zbuduj intrinsics dla kamery RGB – ważne, aby rozmiar był zgodny z obrazem
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_w, height=rgb_h, fx=fx, fy=fy, cx=cx, cy=cy
    )

    cumulative_transform = np.eye(4)
    point_cloud = None
    rgbd_image_prev = None

    # Parametry odometrii
    odo_init = np.eye(4)
    odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    # Zakładamy pewną wartość bazową między kamerami stereo (w metrach)
    baseline = 0.2

    for i in range(num_images):
        # Wczytaj obrazy – rgb pochodzi z osobnego sensora
        rgb = rgb_dataset[i]  # np. kształt: (720, 1280, 3)
        left_img = left_dataset[i]  # np. (800, 1280) – inne rozdzielczości
        right_img = right_dataset[i]

        # Upewnij się, że obrazy stereo są w odcieniach szarości
        if left_img.ndim == 3 and left_img.shape[2] == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img.copy()

        if right_img.ndim == 3 and right_img.shape[2] == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img.copy()

        # Jeśli obrazy stereo mają różne rozmiary, przeskaluj prawy do lewego
        if left_gray.shape != right_gray.shape:
            right_gray = cv2.resize(
                right_gray,
                (left_gray.shape[1], left_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Oblicz dysparycję – StereoBM zwraca wartości przemnożone przez 16
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Aby uniknąć dzielenia przez zero, wartości poniżej pewnego progu zastąp minimalną wartością
        disparity[disparity < 1.0] = 1.0

        # Wyliczenie głębi wg wzoru: depth = fx * baseline / disparity
        depth = fx * baseline / (disparity + 1e-6)  # wynik w metrach

        # UWAGA: mapa głębi wygenerowana z obrazów stereo ma rozmiar lewego obrazu,
        # który może być inny niż rozmiar obrazu RGB. Dlatego przeskalujemy mapę głębi
        # do rozmiaru obrazu RGB.
        if depth.shape[0] != rgb_h or depth.shape[1] != rgb_w:
            depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        # Upewnij się, że obraz kolorowy ma właściwy typ
        color_image = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_image = o3d.geometry.Image(depth.astype(np.float32))

        # Utwórz obiekt RGBDImage
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,  # mapa głębi już w metrach
            depth_trunc=3.0,  # przycinamy głębię, aby wyeliminować bardzo odległe punkty
            convert_rgb_to_intensity=False,
        )

        if i == 0:
            # Pierwsza klatka – inicjalizacja chmury punktów
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic
            )
            rgbd_image_prev = rgbd_image
        else:
            # Oblicz transformację między poprzednią a bieżącą ramką
            success, transformation, info = (
                o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd_image_prev, rgbd_image, intrinsic, odo_init, odo_method
                )
            )
            if success:
                cumulative_transform = cumulative_transform @ transformation
                pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, intrinsic
                )
                pc.transform(cumulative_transform)
                point_cloud += pc
            else:
                print(f"Odometry nie powiodło się dla klatki {i}.")
            rgbd_image_prev = rgbd_image

        print(f"Kumulowana transformacja dla klatki {i}:\n{cumulative_transform}")

    # Wyświetl zintegrowaną chmurę punktów w interaktywnym oknie 3D
    o3d.visualization.draw_geometries([point_cloud])
