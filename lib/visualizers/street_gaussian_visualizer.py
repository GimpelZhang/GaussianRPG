import os
import torchvision
import cv2
import torch
import imageio
import numpy as np

from lib.utils.camera_utils import Camera
from lib.utils.img_utils import visualize_depth_numpy
from lib.config import cfg

class StreetGaussianVisualizer():
    def __init__(self, save_dir):
        
        self.result_dir = save_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.save_video = cfg.render.save_video
        self.save_image = cfg.render.save_image
        
        self.rgbs_gt = []
        self.rgbs = []
        self.rgbs_bkgd = []
        self.rgbs_obj = []
        self.accs_obj = []
        self.depths = []
        self.diffs = []
        self.normals = []
        self.cams = []
        
        self.depth_visualize_func = lambda x: visualize_depth_numpy(x, cmap=cv2.COLORMAP_JET)[0][..., [2, 1, 0]]
        self.diff_visualize_func = lambda x: visualize_depth_numpy(x, cmap=cv2.COLORMAP_TURBO)[0][..., [2, 1, 0]]

            
    def visualize(self, result, camera: Camera):
        self.cams.append(camera.meta['cam'])
        name = camera.image_name
        
        rgb = result['rgb']
        rgb_bkgd = result['rgb_background']
        rgb_obj = result['rgb_object']
        acc_obj = result['acc_object']
        
        if self.save_image:
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, f'{name}_rgb.png'))
            torchvision.utils.save_image(rgb_bkgd, os.path.join(self.result_dir, f'{name}_rgb_bkgd.png'))
            torchvision.utils.save_image(rgb_obj, os.path.join(self.result_dir, f'{name}_rgb_obj.png'))
            torchvision.utils.save_image(acc_obj.float(), os.path.join(self.result_dir, f'{name}_acc_obj.png'))
            torchvision.utils.save_image(camera.original_image[:3], os.path.join(self.result_dir, f'{name}_gt.png'))
    
        if self.save_video:
            rgb_gt = (camera.original_image[:3].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs_gt.append(rgb_gt)
            rgb_bkgd = (rgb_bkgd.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs_bkgd.append(rgb_bkgd)
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)                                            
            self.rgbs.append(rgb)
            rgb_obj = (rgb_obj.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs_obj.append(rgb_obj)
            
            acc_obj = (acc_obj.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.accs_obj.append(acc_obj)

        
        self.visualize_diff(result, camera)
        self.visualize_depth(result, camera)
        self.visualize_normal(result, camera)
    
    def visualize_novel_view(self, result, camera: Camera):
        self.cams.append(camera.meta['cam'])
        id = camera.id
        rgb = result['rgb']
        depth = result['depth']
        depth = depth.permute(1, 2, 0).detach().cpu().numpy() # [H, W, 1]
        # np.save(os.path.join(self.result_dir, f'{id}_depth.npy'), depth)

        if self.save_image:
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, f'{id:06d}_rgb.png'))
            imageio.imwrite(os.path.join(self.result_dir, f'{id:06d}_depth.png'), self.diff_visualize_func(depth))

        if self.save_video:
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs.append(rgb)
            self.depths.append(depth)
                
    def visualize_diff(self, result, camera: Camera):
        name = camera.image_name
        rgb_gt = camera.original_image[:3]
        rgb = result['rgb'].detach().cpu()
        
        if hasattr(camera, 'original_mask'):
            mask = camera.original_mask.bool()
        else:
            mask = torch.ones_like(rgb[0]).bool()
            
        rgb = torch.where(mask, rgb, torch.zeros_like(rgb))
        rgb_gt = torch.where(mask, rgb_gt, torch.zeros_like(rgb_gt))
        
        rgb = rgb.permute(1, 2, 0).numpy() # [H, W, 3]
        rgb_gt = rgb_gt.permute(1, 2, 0).numpy() # [H, W, 3]
        diff = ((rgb - rgb_gt) ** 2).sum(axis=-1, keepdims=True) # [H, W, 1]
        
        if self.save_image:
            imageio.imwrite(os.path.join(self.result_dir, f'{name}_diff.png'), self.diff_visualize_func(diff))
        
        if self.save_video:
            self.diffs.append(diff)

    def visualize_depth(self, result, camera: Camera):
        name = camera.image_name
        depth = result['depth']

        depth = depth.detach().permute(1, 2, 0).detach().cpu().numpy() # [H, W, 1]
        
        if self.save_image:
            imageio.imwrite(os.path.join(self.result_dir, f'{name}_depth.png'), self.diff_visualize_func(depth))

        if self.save_video:
            self.depths.append(depth)
            
    def visualize_normal(self, result, camera: Camera):
        if 'normals' in result.keys():            
            name = camera.image_name
            normals = result['normals'].detach().permute(1, 2, 0) # [H, W, 3]
            
            # transform normal to camera space
            # R_w2c = camera.world_view_transform[:3, :3]
            # normals = torch.matmul(normals, R_w2c.T)
            
            normals = (normals + 1) / 2.0 # to 0 - 1
            normals = (normals.cpu().numpy() * 255).astype(np.uint8)

            if self.save_image:
                imageio.imwrite(os.path.join(self.result_dir, f'{name}_normal.png'), normals)

            if self.save_video:
                self.normals.append(normals)

    def save_video_from_frames(self, frames, name, visualize_func=None):
        if len(frames) == 0:
            return
        
        unqiue_cams = sorted(list(set(self.cams)))
        if len(unqiue_cams) == 1:
            if visualize_func is not None:
                frames = [visualize_func(frame) for frame in frames]
            imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames, fps=cfg.render.fps)
        else:
            concat_cameras = cfg.render.get('concat_cameras', [])
            if len(concat_cameras) == len(unqiue_cams):
                frames_cam_all = []
                for cam in concat_cameras:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]
                    frames_cam_all.append(frames_cam)
                
                frames_cam_len = [len(frames_cam) for frames_cam in frames_cam_all]
                assert len(list(set(frames_cam_len))) == 1, 'all cameras should have same number of frames'
                num_frames = frames_cam_len[0]

                
                frames_concat_all = []
                for i in range(num_frames):
                    frames_concat = []
                    for j in range(len(concat_cameras)):
                        frames_concat.append(frames_cam_all[j][i])
                    frames_concat = np.concatenate(frames_concat, axis=1)
                    frames_concat_all.append(frames_concat)
                
                if visualize_func is not None:
                    frames_concat_all = [visualize_func(frame) for frame in frames_concat_all]    
        
                imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames_concat_all, fps=cfg.render.fps)
            
            else:
                for cam in unqiue_cams:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]
                    
                    if visualize_func is not None:
                        frames_cam = [visualize_func(frame) for frame in frames_cam]
                    
                    imageio.mimwrite(os.path.join(self.result_dir, f'{name}_{str(cam)}.mp4'), frames_cam, fps=cfg.render.fps)

    def summarize(self):                
        if cfg.render.get('save_video', True):
            self.save_video_from_frames(self.rgbs_gt, 'color_gt')
            self.save_video_from_frames(self.rgbs, 'color')
            self.save_video_from_frames(self.rgbs_bkgd, 'color_bkgd')
            self.save_video_from_frames(self.rgbs_obj, 'color_obj')
            self.save_video_from_frames(self.accs_obj, 'acc_obj')
            self.save_video_from_frames(self.depths, 'depth', visualize_func=self.depth_visualize_func)
            self.save_video_from_frames(self.diffs, 'diff', visualize_func=self.diff_visualize_func)

class StreetGaussianVisualizerLite():
    def __init__(self, save_dir):

        self.result_dir = save_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.save_video = cfg.render.save_video
        self.save_image = cfg.render.save_image

        self.rgbs = []
        self.cams = []

    def visualize(self, result, camera: Camera):
        self.cams.append(camera.meta['cam'])
        name = camera.image_name

        rgb = result['rgb']

        if self.save_image:
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, f'{name}_rgb.png'))

        if self.save_video:
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs.append(rgb)

    def visualize_novel_view(self, result, camera: Camera):
        self.cams.append(camera.meta['cam'])
        id = camera.id
        rgb = result['rgb']
        depth = result['depth']
        depth = depth.permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 1]
        # np.save(os.path.join(self.result_dir, f'{id}_depth.npy'), depth)

        if self.save_image:
            torchvision.utils.save_image(rgb, os.path.join(self.result_dir, f'{id:06d}_rgb.png'))

        if self.save_video:
            rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            self.rgbs.append(rgb)

    def save_video_from_frames(self, frames, name, visualize_func=None):
        if len(frames) == 0:
            return

        unqiue_cams = sorted(list(set(self.cams)))
        if len(unqiue_cams) == 1:
            if visualize_func is not None:
                frames = [visualize_func(frame) for frame in frames]
            imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames, fps=cfg.render.fps)
        else:
            concat_cameras = cfg.render.get('concat_cameras', [])
            if len(concat_cameras) == len(unqiue_cams):
                frames_cam_all = []
                for cam in concat_cameras:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]
                    frames_cam_all.append(frames_cam)

                frames_cam_len = [len(frames_cam) for frames_cam in frames_cam_all]
                assert len(list(set(frames_cam_len))) == 1, 'all cameras should have same number of frames'
                num_frames = frames_cam_len[0]

                frames_concat_all = []
                for i in range(num_frames):
                    frames_concat = []
                    for j in range(len(concat_cameras)):
                        frames_concat.append(frames_cam_all[j][i])
                    frames_concat = np.concatenate(frames_concat, axis=1)
                    frames_concat_all.append(frames_concat)

                if visualize_func is not None:
                    frames_concat_all = [visualize_func(frame) for frame in frames_concat_all]

                imageio.mimwrite(os.path.join(self.result_dir, f'{name}.mp4'), frames_concat_all, fps=cfg.render.fps)

            else:
                for cam in unqiue_cams:
                    frames_cam = [frame for frame, c in zip(frames, self.cams) if c == cam]

                    if visualize_func is not None:
                        frames_cam = [visualize_func(frame) for frame in frames_cam]

                    imageio.mimwrite(os.path.join(self.result_dir, f'{name}_{str(cam)}.mp4'), frames_cam,
                                     fps=cfg.render.fps)

    def summarize(self):
        if cfg.render.get('save_video', True):
            self.save_video_from_frames(self.rgbs, 'color')