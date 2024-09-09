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

class CarDynamics(Node):

    def __init__(self):
        super().__init__('car_dynamics')
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

        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, 'cam_pose', 10)
        self.timer_period = 0.02  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.subscription = self.create_subscription(
            TwistStamped,
            'control_cmd',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.sub_sync = self.create_subscription(
            Float64,
            'image_timestamp',
            self.sub_sync_callback,
            10)
        self.sub_sync  # prevent unused variable warning

        print(self.traj_file_path)

    def sub_sync_callback(self, msg):
        print("image_timestamp: ", msg.data)
        if msg.data != self.image_last_stamp:
            self.image_last_stamp = msg.data
            self.sync_lock = False

    def listener_callback(self, msg):
        pass

    def timer_callback(self):
        if not self.sync_lock:
            if self.controller_activated:
                msg = PoseWithCovarianceStamped()
            else:
                if self.idx < self.traj_length:
                    pose = self.traj['frames'][self.idx]
                    self.timestamp = pose['timestamp']
                else:
                    pose = self.traj['frames'][-1]
                    self.timestamp = pose['timestamp'] + (self.idx - self.traj_length + 1) * self.timer_period

                msg = PoseWithCovarianceStamped()
                msg.header.frame_id = 'base_link'
                msg.header.stamp.sec = int(self.timestamp)
                msg.header.stamp.nanosec = int(self.timestamp * 1e9) % 1000000000
                print("car_dynamic: ", self.timestamp)

                msg.pose.pose.position.x = pose['position'][0]
                msg.pose.pose.position.y = pose['position'][1]
                msg.pose.pose.position.z = pose['position'][2]

                Rt = np.array(pose['rotation_matrix'])
                quat = Rotation.from_matrix(Rt).as_quat()

                msg.pose.pose.orientation.x = quat[0]
                msg.pose.pose.orientation.y = quat[1]
                msg.pose.pose.orientation.z = quat[2]
                msg.pose.pose.orientation.w = quat[3]

            self.last_pose_msg = msg
            self.idx = self.idx + 1
            self.sync_iter = self.sync_iter + 1
            # self.publisher_.publish(msg)
            if self.sync_iter >= self.sync_iter_times:
                self.sync_iter = 0
                self.sync_lock = True
        else:
            msg = self.last_pose_msg
        self.publisher_.publish(msg)

    def final_call(self):
        pass

def main(args=None):
    rclpy.init(args=args)

    car_dynamics = CarDynamics()

    rclpy.spin(car_dynamics)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    car_dynamics.final_call()
    car_dynamics.destroy_node()
    rclpy.shutdown()