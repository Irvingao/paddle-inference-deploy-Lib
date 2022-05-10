import cv2
import numpy as np
import time

from Lib.PaddleSeg_Inference_Lib import Paddle_Seg

# --------------------------配置区域--------------------------
infer_img_size = 640        # 自定义模型预测的输入图像尺寸
use_gpu = True              # 是否使用GPU
gpu_memory = 500            # GPU的显存
use_tensorrt = False         # 是否使用TensorRT
precision_mode = "fp16"     # TensorRT精度模式
# 模型文件路径
model_folder_dir = "model/hardnet_seg"
# 类别信息
label_list = ["sidewalk","other","blind_road"]
# -----------------------------------------------------------

def main():
    # 初始化
    # cap = cv2.VideoCapture(0) # USB Camera
    # camera_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # camera_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image = cv2.imread("pic/3_1136.jpg")
    camera_width = 640
    camera_height = 480

    paddle_seg = Paddle_Seg(model_folder_dir=model_folder_dir, infer_img_size=infer_img_size, 
                            use_gpu=use_gpu, gpu_memory=gpu_memory, use_tensorrt=use_tensorrt, 
                            precision_mode=precision_mode, label_list=label_list)
    
    paddle_seg.init(camera_width,camera_height)

    # 预测
    result = paddle_seg.infer(image)
    # 绘制结果
    image, _ = paddle_seg.post_process(image, result)
    
    # cv2.imshow("image", image)
    # cv2.waitKey(1) 
    image=(image*255).astype(np.uint8)
    cv2.imwrite("out.jpg", image)


if __name__ == '__main__':
    main()


