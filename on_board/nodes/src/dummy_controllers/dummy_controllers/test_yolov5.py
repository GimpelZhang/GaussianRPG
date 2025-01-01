#!/usr/bin/env python3

import numpy as np
import cv2
from hobot_dnn import pyeasy_dnn as dnn
import time
import ctypes
import json 


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

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so') 

get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]  
get_Postprocess_result.restype = ctypes.c_char_p  

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)


def bgr2nv12_opencv(image):
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


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


if __name__ == '__main__':
    models = dnn.load('../models/yolov5s_672x672_nv12.bin')
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    # 打印输出 tensor 的属性
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    img_file = cv2.imread('./kite.jpg')
    h, w = get_hw(models[0].inputs[0].properties)
    des_dim = (w, h)
    resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)
    t0 = time.time()
    outputs = models[0].forward(nv12_data)
    t1 = time.time()
    print("inferece time is :", (t1 - t0))

    # 获取结构体信息
    yolov5_postprocess_info = Yolov5PostProcessInfo_t()
    yolov5_postprocess_info.height = h
    yolov5_postprocess_info.width = w
    org_height, org_width = img_file.shape[0:2]
    yolov5_postprocess_info.ori_height = org_height
    yolov5_postprocess_info.ori_width = org_width
    yolov5_postprocess_info.score_threshold = 0.4 
    yolov5_postprocess_info.nms_threshold = 0.45
    yolov5_postprocess_info.nms_top_k = 20
    yolov5_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(outputs[i].properties.layout)
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
        
        libpostprocess.Yolov5doProcess(output_tensors[i], ctypes.pointer(yolov5_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(yolov5_postprocess_info))  
    result_str = result_str.decode('utf-8')  
    t2 = time.time()
    # print("postprocess time is :", (t2 - t1))
    #print(result_str)

    t0 = time.time()
    # draw result
    # 解析JSON字符串  
    data = json.loads(result_str[16:])  

    # 遍历每一个结果  
    for result in data:  
        bbox = result['bbox']  # 矩形框位置信息  
        score = result['score']  # 得分  
        id = result['id']  # id  
        name = result['name']  # 类别名称  
    
        # 打印信息  
        print(f"bbox: {bbox}, score: {score}, id: {id}, name: {name}")

        # 在图片上画出边界框  
        cv2.rectangle(img_file, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)  
    
        # 在边界框上方显示类别名称和得分  
        font = cv2.FONT_HERSHEY_SIMPLEX  
        cv2.putText(img_file, f'{name} {score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), font, 0.5, (0, 255, 0), 1)  
  
    cv2.imwrite('output_image.jpg', img_file)

    t1 = time.time()

    print("draw result time is :", (t1 - t0))

