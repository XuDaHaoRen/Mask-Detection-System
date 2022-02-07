# -*- coding:utf-8 -*-
import cv2
import time

import numpy as np
from PIL import Image
from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.keras_loader import load_keras_model, keras_inference
#界面
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter

model = load_keras_model('models/face_mask_detection.json', 'models/face_mask_detection.hdf5')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(460, 460),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    y_bboxes_output, y_cls_output = keras_inference(model, image_exp)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms. 处理多余的 anchor
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    #绘制 anchor
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path) #如果参数为 0 就读取摄像头
    #cap = cv2.VideoCapture(0) #如果参数为 0 就读取摄像头
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #保存每一帧的数据 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')#视频编码方式
    writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        try:
            ret, img_raw = cap.read()
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            read_frame_stamp = time.time()
            if (ret):
                inference(img_raw,
                          conf_thresh,
                          iou_thresh=0.5,
                          target_shape=(260, 260),
                          draw_result=True,
                          show_result=False)
                cv2.imshow('image', img_raw[:, :, ::-1])
                cv2.waitKey(1)
                inference_stamp = time.time()
                idx += 1
                print("read_frame:%f, infer time:%f" % (read_frame_stamp - start_stamp,
                                                                       inference_stamp - read_frame_stamp))
                # cv2.waitKey() 1：等待一帧后停止； 0：立刻停止
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except:
            print('视频读取结束！')
            cv2.destroyAllWindows()
            break
    cap.release()
    
def image_detect(img_raw):
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) #转换成 RGB 图像
    inference(img_raw,
              conf_thresh=0.5,
              iou_thresh=0.5,
              target_shape=(260, 260),
              draw_result=True,
              show_result=False)
    cv2.imshow('image', img_raw[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#点击打开图片之后的跳转
def open_img():
    #打开本地文件，获取选中的文件地址
    File = filedialog.askopenfilename(parent=top, initialdir="C:/",title='Choose an image.')
    img = cv2.imread(File)
    image_detect(img)
    
    
def open_vid():
    File = filedialog.askopenfilename(parent=top, initialdir="C:/",title='Choose an image.')
    run_on_video(File,'', conf_thresh=0.5)
def open_cam():
    run_on_video(0,'', conf_thresh=0.5)
    



if __name__ == "__main__":
    # 设置检测参数  0表示检测图片  1表示检测视频
    #status=1
    #if status ==0:
        #img = cv2.imread("./img/Mask.jpg")
        #img = cv2.imread("./img/noMask.jpg")
        #image_detect(img)
    #else:
        ## 读入视频测试
        #video_path = './img/test1.mp4'
        #run_on_video(video_path,'', conf_thresh=0.5)   
        
        top = tkinter.Tk()
        top.title("口罩检测")
        top.geometry("500x300")
        btn_img = tkinter.Button(top,text="检测图片",command = lambda:open_img())
        btn_vid = tkinter.Button(top,text="检测视频",command = lambda:open_vid())
        btn_cam = tkinter.Button(top,text="实时检测",command = lambda:open_cam())
        btn_img.pack()
        btn_vid.pack()
        btn_cam.pack()
        top.mainloop()
    
        
        
        
    
        

        
    
        
        
        
        
        
        
        
