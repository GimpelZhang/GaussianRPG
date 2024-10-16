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

from geometry_msgs.msg import PoseWithCovarianceStamped

from scipy.spatial.transform import Rotation
import numpy as np

class GroundTruth(Node):

    def __init__(self):
        super().__init__('ground_truth')
        self.traj_file_path = ''
        self.track_info_path = ''
        self.start_frame = 98  # start frame id of the scene
        self.cipv_id = 6  # the cipv id in the track_info.txt
        if len(sys.argv) < 5:
            self.get_logger().error('arg: traj_path track_info_path start_frame cipv_id')
        else:
            self.traj_file_path = sys.argv[1]
            self.track_info_path = sys.argv[2]
            self.start_frame = int(sys.argv[3])
            self.cipv_id = int(sys.argv[4])

        try:
            with open(self.traj_file_path, 'r', encoding='utf-8') as file:
                traj = json.load(file)
        except FileNotFoundError:
            self.get_logger().error('Traj file not found.')
        except json.JSONDecodeError:
            self.get_logger().error('Parsing traj file failed')
        except Exception as e:
            self.get_logger().error(f"Unkown error：{e}")

        try:
            with open(self.track_info_path, 'r', encoding='utf-8') as file:
                tracklets_str = file.read().splitlines()
                tracklets_str = tracklets_str[1:]
        except FileNotFoundError:
            self.get_logger().error('Traj file not found.')
        except json.JSONDecodeError:
            self.get_logger().error('Parsing traj file failed')
        except Exception as e:
            self.get_logger().error(f"Unkown error：{e}")

        self.sync_iter_times = int(traj['dynamic_freq'] / traj['image_freq'])

        self.tracklets_ls = dict()

        for tracklet in tracklets_str:
            tracklet = tracklet.split()
            if int(tracklet[0]) >= self.start_frame:
                if int(tracklet[1]) == self.cipv_id:
                    frame_id = int(tracklet[0])
                    self.tracklets_ls[frame_id] = [float(t) for t in tracklet[7:10]] # xyz
        # print(self.tracklets_ls)

        self.traj_ls = dict()

        for traj_point in traj['frames']:
            self.traj_ls[traj_point['id']] = np.array(traj_point['ego_pose'])
        # print(self.traj_ls)

        self.publisher_distance = self.create_publisher(Float64, 'cam_2_cipv', 10)

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'cam_pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning


    def transform_pose(self, ego_pose):
        # ego_pose is transformed. After that all ego cars move in x direction
        Rt = ego_pose[:3, :3]
        rot_obj = Rotation.from_matrix(Rt)
        angles = rot_obj.as_euler('xyz')
        m_angles = -angles
        m_Rt = Rotation.from_euler('xyz', m_angles.tolist())
        m_Rt = m_Rt.as_matrix()
        Trans = np.eye(4)
        Trans[:3, :3] = m_Rt

        return Trans @ ego_pose


    def listener_callback(self, msg):
        idx = int(msg.pose.covariance[0])
        if idx % self.sync_iter_times == 0:
            if idx < len(self.traj_ls):
                ego_pose = self.transform_pose(self.traj_ls[idx])
                track_info_idx = idx // self.sync_iter_times + self.start_frame
                track_info = self.tracklets_ls[track_info_idx]
            else:
                ego_pose = self.transform_pose(list(self.traj_ls.values())[-1])
                track_info = list(self.tracklets_ls.values())[-1]
            distance_msg = Float64()
            # After transforming, all ego cars move in x direction
            # so distance = (ego pose in world coord) + (cipv pose in ego coord) - (cam pose in world coord)
            # of course only one direction is considered in the following line:
            distance_msg.data = ego_pose[0, 3] + track_info[0] - (-msg.pose.pose.position.z)
            # print("idx: ", idx)
            # print(" cam to cipv: ", distance_msg.data)
            time_stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
            self.get_logger().info('timestamp: "%s"' % time_stamp)
            self.get_logger().info('GT cam to cipv: "%s"' % distance_msg.data)
            self.publisher_distance.publish(distance_msg)


    def final_call(self):
        pass

def main(args=None):
    rclpy.init(args=args)

    ground_truth = GroundTruth()

    rclpy.spin(ground_truth)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ground_truth.final_call()
    ground_truth.destroy_node()
    rclpy.shutdown()