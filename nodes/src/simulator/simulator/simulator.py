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
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped

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

class MainFrame(Node):

    def __init__(self):
        super().__init__('gaussian_rpg')
        self.publisher_image = self.create_publisher(ROS2_Image, 'front_image', 10)
        self.publisher_pose = self.create_publisher(PoseWithCovarianceStamped, 'cam_pose', 10)
        self.subscription = self.create_subscription(
            TwistStamped,
            'control_cmd',
            self.control_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.timer_period = 0.02
        self.frame_timer = self.create_timer(self.timer_period, self.next_frame)

        self.last_pose_dict = {"position": [0.0, 0.0, 0.0]}
        self.forward_velocity = 0.0
        self.command_brake = 0.0

        self.controller_activated = False
        self.idx = 1
        self.timestamp = 0.0

        try:
            with open(cfg.traj_file_path, 'r', encoding='utf-8') as file:
                self.traj = json.load(file)
        except FileNotFoundError:
            self.get_logger().error('Traj file not found.')
        except json.JSONDecodeError:
            self.get_logger().error('Parsing traj file failed')
        except Exception as e:
            self.get_logger().error(f"Unkown error：{e}")

        self.traj_length = len(self.traj['frames'])
        self.sync_iter_times = int(self.traj['dynamic_freq'] / self.traj['image_freq'])
        self.sync_iter = 0
        self.sync_lock = False
        self.image_last_stamp = -0.1

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
            self.render() # render the first frame here

    def next_frame(self):
        if not self.sync_lock:
            if self.idx < self.traj_length:
                pose = self.traj['frames'][self.idx]
                self.timestamp = pose['timestamp']
            else:
                pose = self.traj['frames'][-1]
                self.timestamp = pose['timestamp'] + (self.idx - self.traj_length + 1) * self.timer_period

            if self.controller_activated:
                velocity_step = self.forward_velocity + self.command_brake * self.timer_period
                if velocity_step < 0:
                    velocity_step = 0
                print("velocity_step: ", velocity_step)
                cam_rot = copy.deepcopy(self.last_pose_dict['rotation_matrix'])
                cam_trans = copy.deepcopy(self.last_pose_dict['position'])
                Rt = np.array(cam_rot).transpose()
                # for cam: -z;
                cam_direction_vector = -Rt[:, 0]

                # update cam position
                update_cam = np.array(cam_trans)
                update_cam = update_cam + velocity_step * self.timer_period * cam_direction_vector
                pose['rotation_matrix'] = cam_rot
                pose['position'] = update_cam.tolist()

            print("car dynamic: ", self.timestamp)
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp.sec = int(self.timestamp)
            msg.header.stamp.nanosec = int(self.timestamp * 1e9) % 1000000000

            msg.pose.pose.position.x = pose['position'][0]
            msg.pose.pose.position.y = pose['position'][1]
            msg.pose.pose.position.z = pose['position'][2]

            Rt = np.array(pose['rotation_matrix'])
            quat = Rotation.from_matrix(Rt).as_quat()

            msg.pose.pose.orientation.x = quat[0]
            msg.pose.pose.orientation.y = quat[1]
            msg.pose.pose.orientation.z = quat[2]
            msg.pose.pose.orientation.w = quat[3]

            self.idx = self.idx + 1
            self.sync_iter = self.sync_iter + 1
            self.publisher_pose.publish(msg)
            self.forward_velocity = np.linalg.norm(np.array(pose['position'])
                                                   - np.array(self.last_pose_dict['position']))/self.timer_period
            self.last_pose_dict = pose
            print("velocity: ", self.forward_velocity)
            # if self.idx > 20:
            #     self.controller_activated = True
            #     print("Brake true.")
            if self.sync_iter >= self.sync_iter_times:
                self.sync_iter = 0
                self.sync_lock = True

                R = np.array(pose['rotation_matrix'])
                T = np.array(pose['position'])
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
                ego_pose = torch.from_numpy(np.array(pose['ego_pose'])).float()
                cam_sample.ego_pose = ego_pose.cuda(0)
                cam_sample.extrinsic = self.cam_orig.extrinsic
                cam_sample.id = self.idx
                cam_sample.meta['frame'] = self.idx
                cam_sample.meta['frame_idx'] = self.idx
                cam_sample.meta['timestamp'] = self.timestamp
                cam_sample.image_name = '000%s_0' % cam_sample.meta['frame']
                self.cam_sample = cam_sample
                self.render()

    def render(self):
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
        self.publisher_image.publish(msg)
        self.sync_lock = False
        # if self.i < self.counter:
        #     self.cam_sample = self.cameras[self.i]
        # else:
        #     self.cam_sample = self.cameras[-1]
        end_time = time.perf_counter()
        # print(f"2执行时间：{end_time - start_time} 秒")

    def control_callback(self, msg):
        if np.abs(msg.twist.linear.x) > 0:
            self.controller_activated = True
            self.command_brake = msg.twist.linear.x
            self.get_logger().info('Brake pressed: ', self.command_brake)

    def final_call(self):
        # self.visualizer.summarize()
        pass

def main(args=None):
    rclpy.init(args=args)

    main_frame = MainFrame()

    rclpy.spin(main_frame)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    main_frame.final_call()
    main_frame.destroy_node()
    rclpy.shutdown()