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
import sys, signal, random
import os, cv2, math
from sensor_msgs.msg import Image as ROS2_Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError

from hobot_dnn import pyeasy_dnn as dnn
import time
import ctypes
import json 

import numpy as np


class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]    

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]

def signal_handler(signal, frame):
    print("\nExiting program")
    sys.exit(0)

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')
        
        # if len(sys.argv) < 2:
        #     self.get_logger().error('arg: yolov5_weights_path')
        #     return
        # else:
        #     self.weights = sys.argv[1]

        self.item_id = 0

        self.publisher_objects = self.create_publisher(PoseArray, 'objects', 10)

        # self.subscription_image = self.create_subscription(
        #     ROS2_Image,
        #     'front_image',
        #     self.image_callback,
        #     10)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # self.subscription_image  # prevent unused variable warning
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 10  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        extr = np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # we directly use the waymo settings here:
        extr[2, 3] = -1.544414504251094167
        self.cam_height = 2.115953541712884078
        intr = np.array(
            [[2.083091212133253975e+03, 0.0, 800.0], [0.0, 2.083091212133253975e+03, 533], [0.0, 0.0, 1.0]])
        self.extrinsics = extr
        self.intrinsics = intr
        self.W = 1600
        self.H = 1066
        self.imgsz = 1600  # inference size (pixels)
        # print("self.imgsz: ", self.imgsz)

        # self.device = select_device(self.device)

        # Load model
        self.models = dnn.load('/app/model/basic/yolov5s_672x672_nv12.bin')

        video_device = self.find_first_usb_camera()
        if video_device is None:
            print("No USB camera found.")
            sys.exit(-1)

        print(f"Opening video device: {video_device}")

        self.cap = cv2.VideoCapture(video_device)
        if(not self.cap.isOpened()):
            exit(-1)
        print("Open usb camera successfully")
        codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
        self.cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # Because the input simulating usb cam is 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        self.libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so') 

        self.get_Postprocess_result = self.libpostprocess.Yolov5PostProcess
        self.get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]  
        self.get_Postprocess_result.restype = ctypes.c_char_p 

        h, w = self.get_hw(self.models[0].inputs[0].properties)
        self.des_dim = (w, h)

        # 获取结构体信息
        self.yolov5_postprocess_info = Yolov5PostProcessInfo_t()
        self.yolov5_postprocess_info.height = h
        self.yolov5_postprocess_info.width = w
        self.yolov5_postprocess_info.ori_height = 1066
        self.yolov5_postprocess_info.ori_width = 1600
        self.yolov5_postprocess_info.score_threshold = 0.4 
        self.yolov5_postprocess_info.nms_threshold = 0.45
        self.yolov5_postprocess_info.nms_top_k = 20
        self.yolov5_postprocess_info.is_pad_resize = 0

    def get_TensorLayout(self, Layout):
        if Layout == "NCHW":
            return int(2)
        else:
            return int(0)


    def bgr2nv12_opencv(self, image):
        height, width = image.shape[0], image.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12


    def get_hw(self, pro):
        if pro.layout == "NCHW":
            return pro.shape[2], pro.shape[3]
        else:
            return pro.shape[1], pro.shape[2]

    def is_usb_camera(self, device):
        try:
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                return False
            cap.release()
            return True
        except Exception:
            return False

    def find_first_usb_camera(self):
        video_devices = [os.path.join('/dev', dev) for dev in os.listdir('/dev') if dev.startswith('video')]
        for dev in video_devices:
            if self.is_usb_camera(dev):
                return dev
        return None


    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        cv2.putText(img, label, c1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        print("左上点的坐标为：(" + str(c1[0]) + "," + str(c1[1]) + ")，右下点的坐标为(" + str(c2[0]) + "," + str(c2[1]) + ")")


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
            u, v, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            # print('目标框', u, v, w, h)
            d1 = self.object_point_world_position(u, v, w, h, p, k)
        distance = 0
        # print('距离', d1)
        if d1[0] <= 0:
            d1[:] = 0
        else:
            distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
        return distance, d1

    
    def timer_callback(self):
        detection_msg = PoseArray()
        detection_msg.header.frame_id = 'base_link'
        
        curr_time = time.time()
        
        # this means the timestamp of detection_msg is the time when the image is captured:
        detection_msg.header.stamp.sec = int(curr_time)
        detection_msg.header.stamp.nanosec = int((curr_time - int(curr_time)) * 1e9)

        _ ,frame = self.cap.read()

        img_file = frame[:1066, :1600, :]

        # img_file = cv2.imread('./kite.jpg')
        resized_data = cv2.resize(img_file, self.des_dim, interpolation=cv2.INTER_AREA)
        nv12_data = self.bgr2nv12_opencv(resized_data)
        # yolov5 detection results saved for debugging:
        rgb_result = img_file.copy()
        t0 = time.time()
        outputs = self.models[0].forward(nv12_data)
        t1 = time.time()
        print("inferece time is :", (t1 - t0))

        output_tensors = (hbDNNTensor_t * len(self.models[0].outputs))()
        for i in range(len(self.models[0].outputs)):
            output_tensors[i].properties.tensorLayout = self.get_TensorLayout(outputs[i].properties.layout)
            # print(output_tensors[i].properties.tensorLayout)
            if (len(outputs[i].properties.scale_data) == 0):
                output_tensors[i].properties.quantiType = 0
                output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            else:
                output_tensors[i].properties.quantiType = 2       
                output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
                
            for j in range(len(outputs[i].properties.shape)):
                output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]
            
            self.libpostprocess.Yolov5doProcess(output_tensors[i], ctypes.pointer(self.yolov5_postprocess_info), i)

        result_str = self.get_Postprocess_result(ctypes.pointer(self.yolov5_postprocess_info))  
        result_str = result_str.decode('utf-8')  
        
        # 解析JSON字符串  
        data = json.loads(result_str[16:])  

        if len(data) > 0:
            print(self.item_id)
            # yolov5 detection results saved for debugging:
            save_path = '/home/sunrise/ros2_ws/nodes/' + '000%s_0' % self.item_id + '.jpg'
            self.item_id += 1

            # 遍历每一个结果  
            for result in data:  
                bbox = result['bbox']  # 矩形框位置信息  
                score = result['score']  # 得分  
                id = result['id']  # id  
                name = result['name']  # 类别名称  
            
                # 打印信息  
                print(f"bbox: {bbox}, score: {score}, id: {id}, name: {name}")

                if not name in ['person', 'chair', 'car', 'truck', 'bicycle', 'motorcycle','bus','traffic light', 'stop sign']:
                    continue

                if float(score) < 0.3:
                    continue

                xywh = np.zeros(4)
                xywh[0] = (bbox[0] + bbox[2]) / 2.0
                xywh[1] = (bbox[1] + bbox[3]) / 2.0
                xywh[2] = bbox[2] - bbox[0]
                xywh[3] = bbox[3] - bbox[1]

                distance, d = self.distance(xywh)
                if d[0] != 0 or d[1] != 0:
                    object_pose = Pose()
                    object_pose.position.x = d[0]
                    object_pose.position.y = d[1]
                    object_pose.position.z = 0.0
                    object_pose.orientation.x = 0.0
                    object_pose.orientation.y = 0.0
                    object_pose.orientation.z = 0.0
                    object_pose.orientation.w = 1.0
                    detection_msg.poses.append(object_pose)
                
                # yolov5 detection results saved for debugging:
                hide_labels = False
                line_thickness = 3
                label = None if hide_labels else name
                if label != None and distance != 0:
                    label = label + ' ' + str('%.1f' % d[0]) + 'm' + str('%.1f' % d[1]) + 'm'
                
                self.plot_one_box(bbox, rgb_result, label=label, line_thickness=line_thickness)

            # yolov5 detection results saved for debugging:
            cv2.imwrite(save_path, rgb_result)
        self.publisher_objects.publish(detection_msg)


    def final_call(self):
        # self.visualizer.summarize()
        pass

def main(args=None):
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init(args=args)

    object_detector = ObjectDetector()

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detector.final_call()
    object_detector.destroy_node()
    rclpy.shutdown()