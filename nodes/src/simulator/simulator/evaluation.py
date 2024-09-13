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
        super().__init__('evaluation')
        self.simulation_time = 20.0
        self.collision_threshold = 2.0
        if len(sys.argv) > 2:
            self.simulation_time = float(sys.argv[1])
            self.collision_threshold = float(sys.argv[2])
        elif len(sys.argv) > 1:
            self.simulation_time = float(sys.argv[1])

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

    def cam_pose_callback(self, msg):
        if msg.header.stamp.sec > self.simulation_time:
            self.get_logger().info('Succeeded. The simulation ends at: "%s" s' % self.simulation_time)
            raise SystemExit()

    def distance_gt_callback(self, msg):
        if msg.data < self.collision_threshold:
            self.get_logger().error('Failed. Collision at: "%s" m' % msg.data)
            raise SystemExit()

    def final_call(self):
        pass

def main(args=None):
    rclpy.init(args=args)

    evaluation = Evaluation()
    try:
        rclpy.spin(evaluation)
    except SystemExit:
        evaluation.get_logger().info(' **Simulation done.** ')
        evaluation.final_call()
        evaluation.destroy_node()
        rclpy.shutdown()