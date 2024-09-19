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
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped, PoseArray, Pose

from scipy.spatial.transform import Rotation

nerf_path = '../'
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

import argparse
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import *
from utils.plots import *
from utils.torch_utils import *

from utils.augmentations import letterbox

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
            return
        except json.JSONDecodeError:
            self.get_logger().error('Parsing traj file failed')
            return
        except Exception as e:
            self.get_logger().error(f"Unkown error：{e}")
            return

        self.traj_length = len(self.traj['frames'])
        self.sync_iter_times = int(self.traj['dynamic_freq'] / self.traj['image_freq'])
        self.sync_iter = 0
        self.sync_lock = False
        self.image_last_stamp = -0.1

        self.separate_perception = cfg.separate_perception
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

            if not self.separate_perception:
                self.publisher_objects = self.create_publisher(PoseArray, 'objects', 10)
                self.weights = cfg.yolov5_weights_path
                self.conf_thres = 0.25  # confidence threshold
                self.iou_thres = 0.45  # NMS IOU threshold
                self.max_det = 10  # maximum detections per image
                self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                self.classes = None  # filter by class: --class 0, or --class 0 2 3
                self.agnostic_nms = False  # class-agnostic NMS
                self.augment = False  # augmented inference
                extr = np.array(
                    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
                extr[2, 3] = -float(self.cam_sample.extrinsic[0, 3]) # ego to cipv
                self.cam_height = float(self.cam_sample.extrinsic[2, 3])
                # extr[2, 3] = -1.544414504251094167
                intr = np.array(
                    [[2.083091212133253975e+03, 0.0, 800.0], [0.0, 2.083091212133253975e+03, 533], [0.0, 0.0, 1.0]])
                self.extrinsics = extr
                self.intrinsics = intr
                self.W = self.cam_sample.image_width
                self.H = self.cam_sample.image_height
                self.imgsz = self.cam_sample.image_width  # inference size (pixels)
                # print("self.imgsz: ", self.imgsz)

                self.device = select_device(self.device)

                # Load model
                self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
                self.stride = max(int(self.model.stride.max()), 32)  # model stride
                self.names = self.model.module.names if hasattr(self.model,
                                                                "module") else self.model.names  # get class names

                # Run inference
                if self.device.type != 'cpu':
                    self.model(
                        torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                            next(self.model.parameters())))

            self.render()  # render the first frame here

    def object_point_world_position(self, u, v, w, h, p, k):
        u1 = u
        v1 = v + h / 2
        # print('关键点坐标：', u1, v1)

        fx = k[0, 0]
        fy = k[1, 1]
        H = self.cam_height
        angle_a = 0
        angle_b = math.atan((v1 - self.H / 2) / fy)
        angle_c = angle_b + angle_a
        # print('angle_b', angle_b)

        depth = (H / np.sin(angle_c)) * math.cos(angle_b)
        # print('depth', depth)

        k_inv = np.linalg.inv(k)
        p_inv = np.linalg.inv(p)
        # print(p_inv)
        point_c = np.array([u1, v1, 1])
        point_c = np.transpose(point_c)
        # print('point_c', point_c)
        # print('k_inv', k_inv)
        c_position = np.matmul(k_inv, depth * point_c)
        # print('c_position', c_position)
        c_position = np.append(c_position, 1)
        c_position = np.transpose(c_position)
        c_position = np.matmul(p_inv, c_position)
        d1 = np.array((c_position[0], -c_position[1]), dtype=float)
        # print('c_position', c_position)
        # d1 = np.array((c_position[2], -c_position[0]), dtype=float)
        return d1

    def distance(self, bbox, xw=5, yw=0.1):
        # print('=' * 50)
        # print('开始测距')
        d1 = np.array((0.0, 0.0), dtype=float)
        p = self.extrinsics
        k = self.intrinsics
        if len(bbox):
            obj_position = []
            u, v, w, h = bbox[1] * self.W, bbox[2] * self.H, bbox[3] * self.W, bbox[4] * self.H
            # print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        # print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1

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
                # print("velocity_step: ", velocity_step)
                # cam_rot = copy.deepcopy(self.last_pose_dict['rotation_matrix'])
                # cam_trans = copy.deepcopy(self.last_pose_dict['position'])
                cam_rot = self.last_pose_dict['rotation_matrix']
                cam_trans = self.last_pose_dict['position']
                # Rt = np.array(cam_rot).transpose()
                # for cam: -z;
                # cam_direction_vector = -Rt[:, 0]
                cam_direction_vector = np.array([0.0, 0.0, -1.0])

                # update cam position
                update_cam = np.array(cam_trans)
                update_cam = update_cam + velocity_step * self.timer_period * cam_direction_vector
                pose['rotation_matrix'] = cam_rot
                pose['position'] = update_cam.tolist()

            # print("car dynamic: ", self.timestamp)
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

            msg.pose.covariance[0] = float(self.idx) # pass the idx info to the ground truth provider.

            self.idx = self.idx + 1
            self.sync_iter = self.sync_iter + 1
            self.publisher_pose.publish(msg)
            self.forward_velocity = np.linalg.norm(np.array(pose['position'])
                                                   - np.array(self.last_pose_dict['position']))/self.timer_period
            self.last_pose_dict = pose
            # print("velocity: ", self.forward_velocity)
            # # for debug:
            # if self.idx > 280:
            #     self.controller_activated = True
            #     self.command_brake = -11.0
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
        # start_time = time.perf_counter()
        # print('image_publisher: ', self.cam_sample.meta['timestamp'])
        result = self.renderer.render_all(self.cam_sample, self.gaussians)
        rgb_result = result['rgb']
        rgb_result = (rgb_result.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # end_time = time.perf_counter()
        # print(f"1执行时间：{end_time - start_time} 秒")
        if self.separate_perception:
            msg = ROS2_Image()
            msg.header.frame_id = 'front'
            msg.header.stamp.sec = int(self.cam_sample.meta['timestamp'])
            msg.header.stamp.nanosec = int(self.cam_sample.meta['timestamp'] * 1e9) % 1000000000
            msg.height = rgb_result.shape[0]
            msg.width = rgb_result.shape[1]
            msg.encoding = 'rgb8'
            msg.step = msg.width * 3
            msg.data = rgb_result.reshape(1, msg.height * msg.step).astype(int).tolist()[0]
            self.publisher_image.publish(msg)

            # end_time = time.perf_counter()
            # print(f"2执行时间：{end_time - start_time} 秒")
        else:
            rgb = letterbox(rgb_result, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
            rgb = rgb.transpose((2, 0, 1))[::-1]
            rgb = np.ascontiguousarray(rgb)  # contiguous
            rgb = torch.from_numpy(rgb).to(self.device)
            rgb = rgb.float()  # uint8 to fp16/32
            rgb /= 255.0  # 0 - 255 to 0.0 - 1.0
            if rgb.ndimension() == 3:
                rgb = rgb.unsqueeze(0)

            # # yolov5 detection results saved for debugging:
            # rgb_result = rgb_result.copy()
            # Inference
            pred = self.model(rgb, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)

            msg = PoseArray()
            msg.header.frame_id = 'base_link'
            msg.header.stamp.sec = int(self.cam_sample.meta['timestamp'])
            msg.header.stamp.nanosec = int(self.cam_sample.meta['timestamp'] * 1e9) % 1000000000

            # Process detections 检测过程
            for i, det in enumerate(pred):  # detections per image
                s = ''
                s += '%gx%g ' % rgb.shape[2:]  # print string 图片形状 eg.640X480
                gn = torch.tensor((self.H, self.W))[[1, 0, 1, 0]]  # normalization gain whwh

                # # yolov5 detection results saved for debugging:
                # save_path = '/home/junchuan/nerf/gaussian_rpg/' + '000%s_0' % self.cam_sample.meta['frame'] + '.jpg'

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
                        if not self.names[int(cls)] in ['person', 'chair', 'car', 'truck', 'bicycle', 'motorcycle',
                                                        'bus',
                                                        'traffic light', 'stop sign']:
                            continue
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        bbox = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]

                        distance, d = self.distance(bbox)
                        if d[0] != 0 or d[1] != 0:
                            object_pose = Pose()
                            object_pose.position.x = d[0]
                            object_pose.position.y = d[1]
                            object_pose.position.z = 0.0
                            object_pose.orientation.x = 0.0
                            object_pose.orientation.y = 0.0
                            object_pose.orientation.z = 0.0
                            object_pose.orientation.w = 1.0
                            msg.poses.append(object_pose)

                        # # yolov5 detection results saved for debugging:
                        # hide_labels = False
                        # hide_conf = False
                        # line_thickness = 3
                        # c = int(cls)  # integer class
                        # label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        # if label != None and distance != 0:
                        #     label = label + ' ' + str('%.1f' % d[0]) + 'm' + str('%.1f' % d[1]) + 'm'
                        #
                        # plot_one_box(xyxy, rgb_result, label=label, color=colors(c, True),
                        #              line_thickness=line_thickness)

                # # yolov5 detection results saved for debugging:
                # cv2.imwrite(save_path, rgb_result)

            self.publisher_objects.publish(msg)
            # end_time = time.perf_counter()
            # print(f"1执行时间：{end_time - start_time} 秒")

        self.sync_lock = False

    def control_callback(self, msg):
        if np.abs(msg.twist.linear.x) > 0:
            self.controller_activated = True
            self.command_brake = msg.twist.linear.x
            self.get_logger().info('Brake pressed: "%s"' % self.command_brake)

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