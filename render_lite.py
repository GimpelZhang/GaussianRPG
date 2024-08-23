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

def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    # Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0
    serializable_array_2d = [x.tolist() for x in Rt]
    camera_entry_before = {
        'id' : id,
        'transform_matrix': serializable_array_2d
    }

    W2C = np.linalg.inv(Rt)
    serializabled = [x.tolist() for x in W2C]
    camera_entry = {
        'id': id,
        'transform_matrix': serializabled
    }
    return camera_entry_before, camera_entry

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
    cfg.render.save_image = False
    cfg.render.save_video = True

    # cfg.render.save_image = True
    # cfg.render.save_video = False
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRendererLite()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizerLite(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        len_cameras = len(cameras)
        json_cams_before = []
        json_cams = []

        for idx in range(len_cameras+5):
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
                x = 0.5  # 举例
                # 从 T1 的旋转矩阵中提取前进方向的单位向量，这里是 z 轴负方向
                direction_vector = -Rt[:, 0]  # 假设物体沿 z 轴负方向前进

                # 计算前进向量
                delta_p = x * direction_vector

                # 更新 T1 的平移向量
                new_translation = fake_idx * delta_p
                print(" new T: ", new_translation)
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
                # cam_sample.meta['timestamp'] = cam_orig.meta['timestamp'] - fake_idx*0.1
                cam_sample.image_name = '000%s_0' % cam_sample.meta['frame']

            print("#### idx: ", idx)
            # print(" camera.R: ", cam_sample.R)
            # print(" camera.R T: ", cam_sample.R.transpose())

            # print(" direction_vector: ", direction_vector)
            print(" camera.T: ", cam_sample.T)
            # print(" new T: ", new_translation)

            result = renderer.render_all(cam_sample, gaussians)
            visualizer.visualize(result, cam_sample)
            _before, _cam = camera_to_JSON(idx, cam_sample)
            json_cams_before.append(_before)
            json_cams.append(_cam)

        json_cams_output_before = {"frames": json_cams_before}
        with open(os.path.join(visualizer.result_dir, "cameras_before.json"), 'w') as file:
            json.dump(json_cams_output_before, file)
        json_cams_output = {"frames": json_cams}
        with open(os.path.join(visualizer.result_dir, "cameras.json"), 'w') as file:
            json.dump(json_cams_output, file)
        # for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
        #     result = renderer.render_all(camera, gaussians)
        #     visualizer.visualize(result, camera)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
