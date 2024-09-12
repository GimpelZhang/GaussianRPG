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
import json

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped

from scipy.spatial.transform import Rotation
import numpy as np

class Evaluation(Node):

    def __init__(self):
        super().__init__('ground_truth')
        self.traj_file_path = ''
        self.controller_activated = False
        self.idx = 0
        self.timestamp = 0.0
        if len(sys.argv) < 2:
            self.get_logger().error('arg: traj_path')
        else:
            self.traj_file_path = sys.argv[1]

        try:
            with open(self.traj_file_path, 'r', encoding='utf-8') as file:
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
        self.last_pose_msg = PoseWithCovarianceStamped()

        self.sub_cam_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            'cam_pose',
            self.cam_pose_callback,
            10)
        self.sub_cam_pose  # prevent unused variable warning

        self.sub_distance_gt = self.create_subscription(
            Float64,
            'cam_2_cipv',
            self.distance_gt_callback,
            10)
        self.sub_distance_gt  # prevent unused variable warning

        print(self.traj_file_path)

    def cam_pose_callback(self, msg):
        pass

    def distance_gt_callback(self, msg):
        pass

    def final_call(self):
        pass

def main(args=None):
    rclpy.init(args=args)

    evaluation = Evaluation()

    rclpy.spin(evaluation)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    evaluation.final_call()
    evaluation.destroy_node()
    rclpy.shutdown()