import cv2
import numpy as np
import yaml
import random
import os
import codecs
import matplotlib.pyplot as plt
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor

import paddleseg.transforms as T
from paddleseg.cvlibs import manager

class DeployYmlConfig:
    def __init__(self, path):
        yml_path = os.path.join(path, "deploy.yaml")
        with codecs.open(yml_path, 'r', 'utf-8') as file:
            try:
                self.dic = yaml.load(file)
            except TypeError:
                self.dic = yaml.load(file, Loader=yaml.FullLoader)
        self._transforms = self._load_transforms(
            self.dic['Deploy']['transforms'])
        self._dir = path

    @property
    def transforms(self):
        return self._transforms

    @property
    def model_file(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params_file(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))
        return T.Compose(transforms)

class Paddle_Seg:
    def __init__(self, model_folder_dir, infer_img_size=224, use_gpu=False, 
                 gpu_memory=500, use_tensorrt=False, precision_mode="fp32", label_list=None):
        self.model_folder_dir = model_folder_dir   # 模型所在路径
        self.infer_img_size = infer_img_size       # 模型预测的输入图像尺寸，Default: 224
        self.use_gpu = use_gpu                     # 是否使用GPU，Default: False
        self.gpu_memory = gpu_memory               # GPU的显存，Default: 500
        self.use_tensorrt = use_tensorrt           # 是否使用TensorRT，Default: False
        self.precision = precision_mode            # TensorRT的precision_mode, Default: "fp16"、"fp32"、"int8"
        self.label_list =  ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle'] if label_list == None else label_list # 类别信息，Default: Cityscapes 21 classes
    
    def init(self,camera_width=640,camera_height=480):
        img = np.zeros(shape=(int(camera_height), int(camera_width),3),dtype="float32")
        self.enlarge_scale, self.narrow_scalse = self.img_config_init(img, self.infer_img_size)
        # 从deploy.yml中读出模型相关信息
        self.cfg = DeployYmlConfig(self.model_folder_dir)
        # 初始化预测模型
        self.predictor = self.predict_config() 

    def img_config_init(self, img, target_size):
        self.im_shape = img.shape
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])
        narrow_scalse_x = float(target_size) / float(self.im_shape[1])
        narrow_scalse_y = float(target_size) / float(self.im_shape[0])
        enlarge_scale_x = float(self.im_shape[1]) / float(target_size)
        enlarge_scale_y = float(self.im_shape[0]) / float(target_size)
        return [enlarge_scale_x, enlarge_scale_y], [narrow_scalse_x, narrow_scalse_y]

    def predict_config(self):
    # ——————————————模型配置、预测相关函数————————————————— #
        # 根据预测部署的实际情况，设置Config
        config = Config()
        # 读取模型文件
        config.set_prog_file(self.cfg.model_file)
        config.set_params_file(self.cfg.params_file)
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
                    use_static = True
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=precision_map[self.precision],
                                            max_batch_size=1, min_subgraph_size=50, 
                                            use_static=use_static, use_calib_mode=use_calib_mode)
                min_input_shape = {"x": [1, 3, 10, 10]}
                max_input_shape = {"x": [1, 3, 1500, 1500]}
                opt_input_shape = {"x": [1, 3, 256, 256]}
                config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)
        print("----------------------------------------------") 
        print("                 RUNNING CONFIG                 ") 
        print("----------------------------------------------") 
        print("Image input size: {}".format(list(self.im_shape))) # 0
        print("Model input size: {}".format([self.infer_img_size, self.infer_img_size, 3])) # 0
        print("Use GPU is: {}".format(config.use_gpu())) # True
        print("GPU device id: {}".format(config.gpu_device_id())) # 0
        print("Init mem size: {}".format(config.memory_pool_init_size_mb())) # 100
        print("Use TensorRT: {}".format(self.use_tensorrt)) # 0
        print("Precision mode: {}".format(precision_map[self.precision])) # 0
        print("enlarge_scale:", self.enlarge_scale)
        print("narrow_scalse:", self.narrow_scalse)
        print("----------------------------------------------") 
        # 可以设置开启IR优化、开启内存优化
        config.switch_ir_optim()
        config.enable_memory_optim()
        predictor = create_predictor(config)
        return predictor

    def resize(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError('image type is not numpy.')
        img = cv2.resize(img, None, None, fx=self.narrow_scalse[0], fy=self.narrow_scalse[1])
        return img

    def preprocess(self, img):
        img = self.resize(img)
        data = np.array([self.cfg.transforms(img)[0]])
        return data

    def predict(self, predictor, data):
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        predictor.run() # 执行Predictor
        results = [] 
        results = output_handle.copy_to_cpu() # 获取输出
        return results

    def infer(self, img):
        data = self.preprocess(img)
        result = self.predict(self.predictor, data)
        return result

    def post_resize(self, img, resize_type):
        if resize_type == 1: # 1:resize到原图大小
            img = cv2.resize(img, None, None, fx=self.enlarge_scale[0], fy=self.enlarge_scale[1])
            # img = cv2.resize(img, (self.im_shape[1], self.im_shape[0]))
        elif resize_type == -1: # -1:resize到模型输入大小
            # img = cv2.resize(img, (self.infer_img_size, self.infer_img_size))
            img = cv2.resize(img, None, None, fx=self.narrow_scalse[0], fy=self.narrow_scalse[1])
        return img

    def decode_segmap(self, mask):
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for idx, label in enumerate(self.label_list): # 选择颜色
            r[mask == idx] = label_colors[label][0]
            g[mask == idx] = label_colors[label][1]
            b[mask == idx] = label_colors[label][2]
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def post_process(self, img, res, resize_type=1):
        mask = np.squeeze(res) # 去掉第一维
        mask = self.decode_segmap(mask) # 将mask合成为fgb图
        mask = self.post_resize(mask, 1)
        img = img / 255 

        img = cv2.addWeighted(img, 1, mask, 0.8, 0, dtype = cv2.CV_32F) # 原图掩膜
        return img, mask 

    def visualize(self, img, mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title('display')
        plt.subplot(1,2,1)
        plt.title('img with predict mask')
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.title('predict mask')
        plt.imshow(mask)
        plt.show()


label_colors = {
        'road': np.array([128, 64, 128]), 'sidewalk': np.array([244, 35, 232]), 'building': np.array([70, 70, 70]), 
        'wall': np.array([244, 35, 232]), 'fence': np.array([128, 64, 128]), 'pole': np.array([244, 35, 232]),
        'traffic_light': np.array([128, 64, 128]), 'traffic_sign': np.array([244, 35, 232]), 'vegetation': np.array([128, 64, 128]), 
        'terrain': np.array([244, 35, 232]), 'sky': np.array([128, 64, 128]), 'person': np.array([244, 35, 232]),
        'rider': np.array([128, 64, 128]), 'car': np.array([244, 35, 232]), 'truck': np.array([128, 64, 128]), 
        'bus': np.array([244, 35, 232]), 'train': np.array([128, 64, 128]), 'motorcycle': np.array([244, 35, 232]),
        'bicycle': np.array([128, 64, 128])
    }

if __name__ == "__main__":
    ###################
    model_folder_dir="../model/hardnet_test"
    infer_img_size=500
    use_gpu=True
    gpu_memory=500
    use_tensorrt=False
    precision_mode="fp16"
    label_list = ["sidewalk"]

    ###################
    paddle_seg = Paddle_Seg(model_folder_dir=model_folder_dir, 
                            infer_img_size=infer_img_size, use_gpu=use_gpu, 
                            gpu_memory=gpu_memory, use_tensorrt=use_tensorrt, 
                            precision_mode=precision_mode, label_list=label_list)
    
    img = cv2.imread("../pic/49.jpg")
    paddle_seg.init(img.shape[1], img.shape[0]) 
    
    res = paddle_seg.infer(img)
    
    img, mask  = paddle_seg.post_process(img, res)

    paddle_seg.visualize(img, mask)

    cv2.imshow("img", img)
    cv2.imshow("mask", mask)

    cv2.waitKey(0)


    
    

    
    
    