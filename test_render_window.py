import torch 
import os
import json
from tqdm import tqdm
import numpy as np
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer, StreetGaussianRendererLite
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer, StreetGaussianVisualizerLite
import time
import copy
from lib.utils.camera_utils import Camera
from scipy.spatial.transform import Rotation
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory():
    # cfg.render.save_image = False
    # cfg.render.save_video = True

    cfg.render.save_image = True
    cfg.render.save_video = False

    # # 创建Tkinter窗口
    # root = tk.Tk()
    # root.title("Image Display")
    #
    # # 设置窗口大小和位置
    # window_width, window_height = 1600, 1066
    # position_x, position_y = 100, 100  # 根据需要调整位置
    # root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    # 设置窗口名称
    window_name = 'Image Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口

    # 设置窗口位置和大小
    # 注意：OpenCV不直接支持设置窗口位置，但可以通过调整窗口属性来间接实现
    cv2.moveWindow(window_name, 10, 10)  # 将窗口移动到屏幕上的(100,100)位置
    cv2.resizeWindow(window_name, 1600, 1066)  # 设置窗口大小为1600x1066
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRendererLite()
        
        # save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        # visualizer = StreetGaussianVisualizerLite(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        len_cameras = len(cameras)

        # for idx in range(90, len_cameras+10):
        for idx in range(len_cameras):
            start_time = time.perf_counter()
            if idx < len_cameras:
                cam_sample = cameras[idx]
            else:
                cam_orig = copy.deepcopy(cameras[-1])
                fake_idx = idx - len_cameras + 1
                # cam_orig.T[0] = cam_orig.T[0] - 0.1*fake_idx
                # cam_orig.T[1] = cam_orig.T[1] + 0.1*fake_idx
                # 物体前进的距离 x
                Rt = cam_orig.R.transpose()
                # r = Rotation.from_matrix(Rt)
                # # 获取欧拉角
                # euler_angles = r.as_euler('xyz', degrees=True)
                # print(" camera xyz angle: ", euler_angles)
                # x = 0.5  # 举例
                x = 0.0  # 举例
                # 从 T1 的旋转矩阵中提取前进方向的单位向量，这里是 z 轴负方向
                # direction_vector = -Rt[:, 0]  # 假设物体沿 z 轴负方向前进
                direction_vector = np.array([0.0, 0.0, -1.0])

                # 计算前进向量
                delta_p = x * direction_vector

                # # 更新 T1 的平移向量
                # new_translation = fake_idx * delta_p
                # print(" new T: ", new_translation)
                cam_orig.T = cam_orig.T + fake_idx * delta_p
                if cam_orig.K.is_cuda:
                    K = cam_orig.K.cpu()
                K_array = K.detach().numpy()
                cam_sample = Camera(
                        id=cam_orig.id,
                        R=cam_orig.R,
                        T=cam_orig.T,
                        FoVx=cam_orig.FoVx,
                        FoVy=cam_orig.FoVy,
                        K=K_array,
                        image=cam_orig.original_image,
                        image_name=cam_orig.image_name,
                        metadata=cam_orig.meta
                )
                cam_sample.ego_pose = cam_orig.ego_pose
                cam_sample.extrinsic = cam_orig.extrinsic
                cam_sample.id = cam_orig.id + fake_idx
                cam_sample.meta['frame'] = cam_orig.meta['frame'] + fake_idx
                cam_sample.meta['frame_idx'] = cam_orig.meta['frame_idx'] + fake_idx
                cam_sample.meta['timestamp'] = cam_orig.meta['timestamp'] - fake_idx*0.1
                cam_sample.image_name = '000%s_0' % cam_sample.meta['frame']

            # print("#### idx: ", idx)
            # # print(" camera.R: ", cam_sample.R)
            # # print(" camera.R T: ", cam_sample.R.transpose())
            #
            # # print(" direction_vector: ", direction_vector)
            # print(" camera.T: ", cam_sample.T)
            # # print(" new T: ", new_translation)
            # print(" camera.timestamp: ", cam_sample.meta['timestamp'])

            result = renderer.render_all(cam_sample, gaussians)['rgb']
            rgb = (result.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # 显示图片
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, image_bgr)

            # 按任意键继续，或者等待1ms
            cv2.waitKey(1)
            end_time = time.perf_counter()
            print(f" running time：{end_time - start_time}秒")
        # 销毁所有窗口
        cv2.destroyAllWindows()
            # visualizer.visualize(result, cam_sample)

            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
