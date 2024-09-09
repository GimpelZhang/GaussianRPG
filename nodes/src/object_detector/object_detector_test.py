# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import os
from sensor_msgs.msg import Image as ROS2_Image
import cv2

nerf_path = '/home/junchuan/nerf/street_gaussians'
print(nerf_path)
if nerf_path not in sys.path:
    sys.path.append(nerf_path)

local_path = '/home/junchuan/nerf/street_gaussians/nodes/src/object_detector/object_detector'
sys.path.append(local_path)

# sys.path.append('/home/junchuan/miniconda3/envs/street-gaussian/lib/python3.8/site-packages')
#
import torch
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

import argparse
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import *
from utils.plots import *
from utils.torch_utils import *

from utils.augmentations import letterbox


class ObjectDetector():
    def __init__(self):
        self.i = 0
        self.weights = 'weights/yolov5s.pt'
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 10  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference

        cfg.mode = 'trajectory'
        cfg.render.save_image = True
        cfg.render.save_video = False

        # run once
        t0 = time.time()

        with torch.no_grad():
            dataset = Dataset()
            self.gaussians = StreetGaussianModel(dataset.scene_info.metadata)

            scene = Scene(gaussians=self.gaussians, dataset=dataset)
            self.renderer = StreetGaussianRendererLite()

            save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
            self.visualizer = StreetGaussianVisualizerLite(save_dir)

            train_cameras = scene.getTrainCameras()
            test_cameras = scene.getTestCameras()
            self.cameras = train_cameras + test_cameras
            self.cameras = list(sorted(self.cameras, key=lambda x: x.id))
            self.counter = len(self.cameras)
            self.cam_sample = self.cameras[self.i]
            # self.visualizer.summarize()
            extr = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            # extr[2, 3] = -self.cam_sample.extrinsic[0, 3]
            extr[2, 3] = -1.544414504251094167
            intr = np.array(
                [[2.083091212133253975e+03, 0.0, 800.0], [0.0, 2.083091212133253975e+03, 533], [0.0, 0.0, 1.0]])
            self.extrinsics = extr
            self.intrinsics = intr
            self.W = self.cam_sample.image_width
            self.H = self.cam_sample.image_height
            self.imgsz = self.cam_sample.image_width  # inference size (pixels)

            self.device = select_device(self.device)

            # Load model
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = max(int(self.model.stride.max()), 32)  # model stride
            self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names  # get class names

            # Run inference
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        print('关键点坐标：', u1, v1)

        fx = k[0, 0]
        fy = k[1, 1]
        H = 2.115953541712884078
        angle_a = 0
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        print('point_c', point_c)
        print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        # print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], -c_position[1]), dtype=float)
        print('c_position', c_position)
        # d1 = np.array((c_position[2], -c_position[0]), dtype=float)
        return d1

    def distance(self, kuang, xw=5, yw=0.1):
        print('=' * 50)
        print('开始测距')
        p = self.extrinsics
        k = self.intrinsics
        if len(kuang):
            obj_position = []
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1

    def timer_callback(self):
        start_time = time.perf_counter()
        result = self.renderer.render_all(self.cam_sample, self.gaussians)
        # self.visualizer.visualize(result, self.cam_sample)
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        rgb_result = result['rgb']

        rgb_result = (rgb_result.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        rgb = letterbox(rgb_result, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        rgb = rgb.transpose((2, 0, 1))[::-1]
        # rgb = rgb.transpose((2, 0, 1))
        rgb = np.ascontiguousarray(rgb)  # contiguous
        rgb = torch.from_numpy(rgb).to(self.device)
        rgb = rgb.float()  # uint8 to fp16/32
        rgb /= 255.0  # 0 - 255 to 0.0 - 1.0
        if rgb.ndimension() == 3:
            rgb = rgb.unsqueeze(0)
        rgb_result = rgb_result.copy()
        # Inference
        # t1 = time_synchronized()
        pred = self.model(rgb, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # t2 = time_synchronized()

        # Process detections 检测过程
        for i, det in enumerate(pred):  # detections per image
            s = ''
            s += '%gx%g ' % rgb.shape[2:]  # print string 图片形状 eg.640X480
            gn = torch.tensor((self.H, self.W))[[1, 0, 1, 0]]  # normalization gain whwh
            save_path = '/home/junchuan/nerf/street_gaussians/' + '000%s_0' % self.cam_sample.meta['frame'] + '.jpg'

            print(save_path)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(rgb.shape[2:], det[:, :4], (self.H, self.W, 3)).round()

                # Print results
                for c in det[:, -1].unique():
                    if not self.names[int(c)] in ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus',
                                             'traffic light', 'stop sign']:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if not self.names[int(cls)] in ['person', 'chair', 'car', 'truck', 'bicycle', 'motorcycle', 'bus',
                                               'traffic light', 'stop sign']:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]

                    distance, d = self.distance(kuang)

                    hide_labels = False
                    hide_conf = False
                    line_thickness = 3
                    c = int(cls)  # integer class
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    if label != None and distance != 0:
                        label = label + ' ' + str('%.1f' % d[0]) + 'm' + str('%.1f' % d[1]) + 'm'

                    plot_one_box(xyxy, rgb_result, label=label, color=colors(c, True), line_thickness=line_thickness)

            cv2.imwrite(save_path, rgb_result)

        end_time = time.perf_counter()
        print(f"1执行时间：{end_time - start_time} 秒")

        self.i += 1
        if self.i < self.counter:
            self.cam_sample = self.cameras[self.i]
        else:
            self.cam_sample = self.cameras[-1]
        end_time = time.perf_counter()
        print(f"2执行时间：{end_time - start_time} 秒")

    def final_call(self):
        # self.visualizer.summarize()
        pass

def main(args=None):
    object_detector = ObjectDetector()
    object_detector.timer_callback()
    object_detector.timer_callback()
    object_detector.timer_callback()
    object_detector.timer_callback()

main()