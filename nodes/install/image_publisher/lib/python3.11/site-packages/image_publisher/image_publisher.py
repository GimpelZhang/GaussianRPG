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

nerf_path = '/home/junchuan/nerf/street_gaussians'
print(nerf_path)
if nerf_path not in sys.path:
    sys.path.append(nerf_path)

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

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(ROS2_Image, 'front_image', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        cfg.mode = 'trajectory'
        cfg.render.save_image = True
        cfg.render.save_video = False

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

    def timer_callback(self):
        start_time = time.perf_counter()
        result = self.renderer.render_all(self.cam_sample, self.gaussians)
        # self.visualizer.visualize(result, self.cam_sample)
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        rgb = result['rgb']
        rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        end_time = time.perf_counter()
        print(f"1执行时间：{end_time - start_time} 秒")

        msg = ROS2_Image()
        msg.header.frame_id = 'front'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = rgb.shape[0]
        msg.width = rgb.shape[1]
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3

        # 直接使用astype转换数据类型，避免使用tolist()
        # msg.data = rgb.reshape(1, step_size).astype(int).tolist()[0]
        msg.data = rgb.reshape(1, msg.height * msg.step).astype(int).tolist()[0]
        self.publisher_.publish(msg)
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
    rclpy.init(args=args)

    image_publisher = ImagePublisher()

    rclpy.spin(image_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_publisher.final_call()
    image_publisher.destroy_node()
    rclpy.shutdown()