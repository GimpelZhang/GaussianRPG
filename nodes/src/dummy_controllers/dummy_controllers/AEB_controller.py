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

import sys
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped, PoseArray

class Controller(Node):

    def __init__(self):
        super().__init__('aeb_controller')
        self.brake_distance = 24.0
        if len(sys.argv) < 2:
            self.get_logger().error('arg: brake_distance')
        else:
            self.brake_distance = float(sys.argv[1])

        self.publisher_command = self.create_publisher(TwistStamped, 'control_cmd', 10)

        self.subscription_perception = self.create_subscription(
            PoseArray,
            'objects',
            self.perception_callback,
            10)
        self.subscription_perception  # prevent unused variable warning

        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            'cam_pose',
            self.pose_callback,
            10)
        self.subscription_pose  # prevent unused variable warning

        self.cipv_lon_dist = 10000.0
        self.cipv_lat_range = 1.2 # meter
        self.last_pose_msg = PoseWithCovarianceStamped()
        self.lon_velocity = 0.0

    def perception_callback(self, msg):
        cipv_candidate = list()
        if len(msg.poses) > 0:
            for pose in msg.poses:
                if pose.position.y < self.cipv_lat_range and pose.position.y > -self.cipv_lat_range:
                    cipv_candidate.append(pose.position.x)
        if len(cipv_candidate)>0:
            self.cipv_lon_dist = min(cipv_candidate)

        # print("cipv: ", self.cipv_lon_dist)
        time_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        self.get_logger().info('timestamp: "%s"' % time_stamp)
        self.get_logger().info('cipv: "%s"' % self.cipv_lon_dist)

        command_msg = TwistStamped()
        command_msg.header = msg.header
        command_msg.twist.linear.x = 0.0
        command_msg.twist.linear.y = 0.0
        command_msg.twist.linear.z = 0.0
        command_msg.twist.angular.x = 0.0
        command_msg.twist.angular.y = 0.0
        command_msg.twist.angular.z = 0.0

        if self.cipv_lon_dist < self.brake_distance:
            if self.lon_velocity > 2:
                command_msg.twist.linear.x = -13.5  # m/s^2
            elif self.lon_velocity > 1:
                command_msg.twist.linear.x = -10.0
            elif self.lon_velocity > 0.001:
                command_msg.twist.linear.x = -8.0
            else:
                command_msg.twist.linear.x = 0.0

        # print("command: ", command_msg.twist.linear.x)
        self.get_logger().info('command output: "%s"' % command_msg.twist.linear.x)
        self.publisher_command.publish(command_msg)


    def pose_callback(self, msg):
        last_stamp = float(self.last_pose_msg.header.stamp.sec) + float(self.last_pose_msg.header.stamp.nanosec)/1e9
        current_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)/1e9
        if current_stamp - last_stamp >0:
            # here we just take the cam is moving along -z as granted:
            self.lon_velocity = ((-msg.pose.pose.position.z - (-self.last_pose_msg.pose.pose.position.z))/
                                 (current_stamp - last_stamp))
            self.last_pose_msg = msg

    def final_call(self):
        # self.visualizer.summarize()
        pass

def main(args=None):
    rclpy.init(args=args)

    aeb_controller = Controller()

    rclpy.spin(aeb_controller)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    aeb_controller.final_call()
    aeb_controller.destroy_node()
    rclpy.shutdown()