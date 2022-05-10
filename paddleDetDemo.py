import cv2
import numpy as np
import time

from Lib.PaddleDetection_Inference_Lib import Paddle_Detection

# --------------------------配置区域--------------------------
infer_img_size = 320        # 自定义模型预测的输入图像尺寸
use_model_img_size = True   #　是否使用模型默认输入图像尺寸，默认为True
use_gpu = True              # 是否使用GPU
gpu_memory = 500           # GPU的显存
use_tensorrt = False        # 是否使用TensorRT
precision_mode = "fp16"
batch_size = 1
filter_mode = False
# 模型文件路径
model_folder_dir = "model/ppyolo_tiny_650e_coco"
# -----------------------------------------------------------

def main():
    # 初始化
    cap = cv2.VideoCapture(0) # USB Camera
    camera_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    paddle_infer = Paddle_Detection(model_folder_dir=model_folder_dir, use_model_img_size=use_model_img_size, 
                                    infer_img_size=infer_img_size, use_gpu=use_gpu, filter_mode=filter_mode, 
                                    gpu_memory=gpu_memory, use_tensorrt=use_tensorrt, precision=precision_mode,
                                    batch_size=batch_size)
    paddle_infer.init(camera_width,camera_height)
    while True:
        # _, image = cap.read()
        # 预测
        # result = paddle_infer.infer(image)
        
        img_list = []
        for i in range(batch_size):
            _, image = cap.read()
            img_list.append(image)
        start = time.time()
        result = paddle_infer.batch_infer(img_list)

        # 绘制结果
        for i in range(batch_size):
            image = paddle_infer.draw_bbox_image(image, result[0], threshold=0.3)
        # image, object_list = paddle_infer.object_filter_show(image, result[0], threshold=0.3)
        cv2.imshow("image", image)


        # print(object_list)
        print("FPS:", 1/(time.time()-start)*batch_size, "/s")
        
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    cap.release()

if __name__ == '__main__':
    main()
