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
from std_msgs.msg import String, Float64
import sys
import os
from sensor_msgs.msg import Image as ROS2_Image
from geometry_msgs.msg import PoseWithCovarianceStamped

from scipy.spatial.transform import Rotation

nerf_path = '../'
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
        print("argv len: ", len(sys.argv))
        self.publisher_ = self.create_publisher(ROS2_Image, 'front_image', 10)
        self.publisher_flag = self.create_publisher(Float64, 'image_timestamp', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.sync_iter_times = 5
        self.sync_iter = 0

        self.last_pose_nanosec = 0

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'cam_pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

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
            self.cam_sample = self.cameras[0]
            self.cam_orig = self.cameras[0] # use the first camera as the original camera
            # self.visualizer.summarize()

    def listener_callback(self, msg):
        print("listen: ", msg.header.stamp)
        if msg.header.stamp.nanosec != self.last_pose_nanosec:
            self.last_pose_nanosec = msg.header.stamp.nanosec
            if self.sync_iter < self.sync_iter_times-1:
                self.sync_iter = self.sync_iter + 1
            else:
                self.sync_iter = 0
                quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                                       msg.pose.pose.orientation.w])
                R = Rotation.from_quat(quaternion).as_matrix()
                T = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                K_array = None
                if self.cam_orig.K.is_cuda:
                    K = self.cam_orig.K.cpu()
                    K_array = K.detach().numpy()
                cam_sample = Camera(
                    id=self.cam_orig.id,
                    R=R,
                    T=T,
                    FoVx=self.cam_orig.FoVx,
                    FoVy=self.cam_orig.FoVy,
                    K=K_array,
                    image=self.cam_orig.original_image,
                    image_name=self.cam_orig.image_name,
                    metadata=self.cam_orig.meta
                )
                cam_sample.ego_pose = self.cam_orig.ego_pose
                cam_sample.extrinsic = self.cam_orig.extrinsic
                cam_sample.id = self.i
                cam_sample.meta['frame'] = self.i
                cam_sample.meta['frame_idx'] = self.i
                cam_sample.meta['timestamp'] = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9
                cam_sample.image_name = '000%s_0' % cam_sample.meta['frame']
                self.cam_sample = cam_sample

                start_time = time.perf_counter()
                print('image_publisher: ', self.cam_sample.meta['timestamp'])
                result = self.renderer.render_all(self.cam_sample, self.gaussians)
                # self.visualizer.visualize(result, self.cam_sample)
                # msg = String()
                # msg.data = 'Hello World: %d' % self.i
                # self.publisher_.publish(msg)
                # self.get_logger().info('Publishing: "%s"' % msg.data)
                rgb = result['rgb']
                rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                end_time = time.perf_counter()
                # print(f"1执行时间：{end_time - start_time} 秒")

                msg = ROS2_Image()
                msg.header.frame_id = 'front'
                msg.header.stamp.sec = int(self.cam_sample.meta['timestamp'])
                msg.header.stamp.nanosec = int(self.cam_sample.meta['timestamp'] * 1e9) % 1000000000
                msg.height = rgb.shape[0]
                msg.width = rgb.shape[1]
                msg.encoding = 'rgb8'
                msg.step = msg.width * 3

                # 直接使用astype转换数据类型，避免使用tolist()
                # msg.data = rgb.reshape(1, step_size).astype(int).tolist()[0]
                msg.data = rgb.reshape(1, msg.height * msg.step).astype(int).tolist()[0]
                self.publisher_.publish(msg)
                timestamp_msg = Float64()
                timestamp_msg.data = float(self.cam_sample.meta['timestamp'])
                self.publisher_flag.publish(timestamp_msg)
                self.i += 1
                # if self.i < self.counter:
                #     self.cam_sample = self.cameras[self.i]
                # else:
                #     self.cam_sample = self.cameras[-1]
                end_time = time.perf_counter()
                # print(f"2执行时间：{end_time - start_time} 秒")

    def timer_callback(self):
        # start_time = time.perf_counter()
        # print('image_publisher: ', self.cam_sample.meta['timestamp'])
        # result = self.renderer.render_all(self.cam_sample, self.gaussians)
        # # self.visualizer.visualize(result, self.cam_sample)
        # # msg = String()
        # # msg.data = 'Hello World: %d' % self.i
        # # self.publisher_.publish(msg)
        # # self.get_logger().info('Publishing: "%s"' % msg.data)
        # rgb = result['rgb']
        # rgb = (rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # end_time = time.perf_counter()
        # # print(f"1执行时间：{end_time - start_time} 秒")
        #
        # msg = ROS2_Image()
        # msg.header.frame_id = 'front'
        # msg.header.stamp.sec = int(self.cam_sample.meta['timestamp'])
        # msg.header.stamp.nanosec = int(self.cam_sample.meta['timestamp'] * 1e9) % 1000000000
        # msg.height = rgb.shape[0]
        # msg.width = rgb.shape[1]
        # msg.encoding = 'rgb8'
        # msg.step = msg.width * 3
        #
        # # 直接使用astype转换数据类型，避免使用tolist()
        # # msg.data = rgb.reshape(1, step_size).astype(int).tolist()[0]
        # msg.data = rgb.reshape(1, msg.height * msg.step).astype(int).tolist()[0]
        # self.publisher_.publish(msg)
        # timestamp_msg = Float64()
        # timestamp_msg.data = float(self.cam_sample.meta['timestamp'])
        # self.publisher_flag.publish(timestamp_msg)
        # self.i += 1
        # # if self.i < self.counter:
        # #     self.cam_sample = self.cameras[self.i]
        # # else:
        # #     self.cam_sample = self.cameras[-1]
        # end_time = time.perf_counter()
        # # print(f"2执行时间：{end_time - start_time} 秒")
        pass

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