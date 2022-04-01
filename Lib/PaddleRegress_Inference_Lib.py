import os
import cv2
import numpy as np
import random
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor

class Paddle_Regress:
    def __init__(self, model_folder_dir, use_model_img_size=True, 
                 infer_img_size=224, use_gpu = False, gpu_memory = 500, 
                 use_tensorrt = False, precision="fp32", 
                 filter_mode = False, filter_range=10, filter_rate=5):
        self.model_folder_dir = model_folder_dir
        self.use_model_img_size = use_model_img_size
        self.infer_img_size = infer_img_size       # 模型预测的输入图像尺寸
        self.use_gpu = use_gpu                     # 是否使用GPU，默认False
        self.gpu_memory = gpu_memory               # GPU的显存，默认500
        self.use_tensorrt = use_tensorrt           # 是否使用TensorRT，默认False
        self.precision = precision                 # TensorRT的precision_mode（"int8"、"fp16"、"float32"）
        self.model_target_size = (224, 224)
    
    def init(self,camera_width=640,camera_height=480):
        img = np.zeros(shape=(int(camera_height), int(camera_width),3),dtype="float32")
        # 读取模型文件
        self.get_model_path()
        # 初始化预测模型
        self.predictor = self.predict_config() 
        if self.use_model_img_size == True: 
            target_size = self.model_target_size[0]
        else:
            target_size = self.infer_img_size
        # 参数初始化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean, self.std, self.im_scale, \
        self.scale_factor, self.infer_im_shape = self.img_config_init(img, target_size, mean, std)
        # mask 初始化
        self.img_mask = np.zeros(img.shape, dtype=np.uint8)
        self.img_mask[:200, :] =255   # 高起始:高结束， 长起始:长结束  ,且左上角为起始点
                

    # ————————————————图像预处理函数———————————————— #
    def img_config_init(self, img, target_size, mean, std):
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        scale_factor = np.array([self.infer_img_size * 1. / img.shape[0], self.infer_img_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
        infer_im_shape = np.array([self.infer_img_size, self.infer_img_size]).reshape((1, 2)).astype(np.float32)
        return mean, std, [im_scale_x, im_scale_y], scale_factor, infer_im_shape

    def resize(self, img):
        """resize to target size"""
        if not isinstance(img, np.ndarray):
            raise TypeError('image type is not numpy.')
        img = cv2.resize(img, None, None, fx=self.im_scale[0], fy=self.im_scale[1])
        return img

    def normalize(self, img):
        img = img / 255.0
        img -= self.mean
        img /= self.std
        return img

    def mask(self, img):
        masked_img = cv2.bitwise_and(img, self.img_mask)
        return masked_img

    def preprocess(self, img):
        img = self.resize(img)
        img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
        img = self.normalize(img)
        img = img.transpose((2, 0, 1))  # hwc -> chw
        return img[np.newaxis, :]

    def get_model_path(self):
        '''
        function: get model and config file path
        param {None}
        return {None}
        '''
        if os.path.isdir(self.model_folder_dir):
            for file in os.listdir(self.model_folder_dir):
                file_type = os.path.splitext(file)[1]
                if file_type == ".pdmodel":
                    self.model_file = os.path.join(self.model_folder_dir, file)
                elif file_type == ".pdiparams":
                    self.params_file = os.path.join(self.model_folder_dir, file)
        else:
            raise Exception("It is wrong model path written. Please check your model dir and rerun the program.")

    # ——————————————模型配置、预测相关函数————————————————— #
    def predict_config(self):
        # 根据预测部署的实际情况，设置Config
        config = Config()
        # 读取模型文件
        config.set_prog_file(self.model_file)
        config.set_params_file(self.params_file)
        precision_map = {
                    "int8": PrecisionType.Int8,
                    "fp16": PrecisionType.Half,
                    "fp32": PrecisionType.Float32}
        if self.use_gpu == True:
            config.enable_use_gpu(self.gpu_memory, 0)
            if self.use_tensorrt == True:
                if self.precision == "int8":
                    use_calib_mode = True
                    use_static = True
                if self.precision == "fp16":
                    use_calib_mode = False
                    use_static = True
                else:
                    use_calib_mode = False
                    use_static = False
                
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=precision_map[self.precision],
                                            max_batch_size=1, min_subgraph_size=self.min_subgraph_size, 
                                            use_static=use_static, use_calib_mode=use_calib_mode)
        print("----------------------------------------------") 
        print("                 RUNNING CONFIG                 ") 
        print("----------------------------------------------") 
        print("Model input size: {}".format([self.infer_img_size, self.infer_img_size, 3])) # 0
        print("Use GPU is: {}".format(config.use_gpu())) # True
        print("GPU device id is: {}".format(config.gpu_device_id())) # 0
        print("Init mem size is: {}".format(config.memory_pool_init_size_mb())) # 100
        print("Use TensorRT: {}".format(self.use_tensorrt)) # 0
        print("Precision mode: {}".format(precision_map[self.precision])) # 0
        print("----------------------------------------------") 
        # 可以设置开启IR优化、开启内存优化
        config.switch_ir_optim()
        config.enable_memory_optim()
        predictor = create_predictor(config)
        return predictor

    def predict(self, predictor, img):
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(img.shape)
        input_tensor.copy_from_cpu(img.copy())
        # 执行Predictor
        predictor.run()
        # 获取输出
        results = []
        # 获取输出
        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu()
        return output_data

    def infer(self, img):
        # 预处理
        img = self.mask(img)
        img = self.preprocess(img)
        # 预测
        result = self.predict(self.predictor, img)
        return result