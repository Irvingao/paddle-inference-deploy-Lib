import cv2
import numpy as np
import codecs
import yaml
import random
import os
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor

class Paddle_Detection:
    def __init__(self, model_folder_dir, use_model_img_size=True, 
                 infer_img_size=224, use_gpu = False, gpu_memory = 500, 
                 use_tensorrt = False, precision="fp32", batch_size=1,
                 filter_mode = False, filter_range=10, filter_rate=5):
        self.model_folder_dir = model_folder_dir
        self.use_model_img_size = use_model_img_size
        self.infer_img_size = infer_img_size       # 模型预测的输入图像尺寸
        self.use_gpu = use_gpu                     # 是否使用GPU，默认False
        self.gpu_memory = gpu_memory               # GPU的显存，默认500
        self.use_tensorrt = use_tensorrt           # 是否使用TensorRT，默认False
        self.precision = precision                 # TensorRT的precision_mode（"int8"、"fp16"、"float32"）
        self.batch_size = batch_size
        self.filter_mode = filter_mode             # 滤波模式，通过多帧检测防止误识别，默认关闭
        self.filter_range = filter_range           # 滤波器的范围，默认10        eg: 10：在10帧的范围内滤波
        self.filter_rate = filter_rate             # 有filter_rate帧预测出物体则认为是真，默认5   eg：5：有5帧都存在则为真

    def get_model_path(self):
        '''
        function: get model and config file path
        param {None}
        return {None}
        '''
        if os.path.isdir(self.model_folder_dir):
            self.cfg_yml_folder = self.model_folder_dir
            for file in os.listdir(self.model_folder_dir):
                file_type = os.path.splitext(file)[1]
                if file_type == ".pdmodel":
                    self.model_file = os.path.join(self.model_folder_dir, file)
                elif file_type == ".pdiparams":
                    self.params_file = os.path.join(self.model_folder_dir, file)
                elif file_type == ".yml":
                    self.infer_cfg_file = os.path.join(self.model_folder_dir, file)
        else:
            raise Exception("It is wrong model path written. Please check your model dir and rerun the program.")

    def init(self,camera_width=640,camera_height=480):
        img = np.zeros(shape=(int(camera_height), int(camera_width),3),dtype="float32")
        # 读取模型文件
        self.get_model_path()
        # 从infer_cfg.yml中读出模型相关信息
        mean, std = self.infer_yml_config()
        # 初始化预测模型
        self.predictor = self.predict_config() 
        if self.use_model_img_size == True: 
            target_size = self.model_target_size[0]
        else:
            target_size = self.infer_img_size
        self.mean, self.std, self.im_scale, \
        self.scale_factor, self.infer_im_shape = self.img_config_init(img, target_size, mean, std)
        # 初始化batch infer的infer_im_shape
        self.infer_im_shape = np.concatenate([self.infer_im_shape for i in range(self.batch_size)],0)
        
        # 滤波模式：初始化检测滤波器
        if self.filter_mode:
            self.object_filter_list = []
            for i in range(len(self.label_list)):
                object_filter = [ 0 for j in range(self.filter_range)]
                self.object_filter_list.append(object_filter)

    # ————————————————读取yml参数信息———————————————— #
    def infer_yml_config(self):
        with codecs.open(self.infer_cfg_file, 'r', 'utf-8') as file:
            try:
                yaml_reader = yaml.load(file)
            except TypeError:
                yaml_reader = yaml.load(file, Loader=yaml.FullLoader)
        self.label_list = yaml_reader['label_list']
        self.color_list = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(self.label_list))]
        self.min_subgraph_size = yaml_reader['min_subgraph_size']
        mean = yaml_reader['Preprocess'][1]['mean'] # [0.485, 0.456, 0.406]
        std = yaml_reader['Preprocess'][1]['std'] # = [0.229, 0.224, 0.225]
        self.model_target_size = yaml_reader['Preprocess'][0]['target_size'] # = [0.229, 0.224, 0.225]
        print("The target size of the model is :", self.model_target_size, \
            "Please make sure ur target_size is same as it when uses TensorRT, ignoring it if not.")
        # key = input()
        return mean, std

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

    def preprocess(self, img):
        img = self.resize(img)
        img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
        img = self.normalize(img)
        img = img.transpose((2, 0, 1))  # hwc -> chw
        return img[np.newaxis, :]

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
                    use_static = True
                
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
        print("Batch size is: {}".format(self.batch_size))
        if self.use_tensorrt == True:
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
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)
            input_tensor.copy_from_cpu(img[i].copy())
        # 执行Predictor
        predictor.run()
        # 获取输出
        results = []
        # 获取输出
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results
    
    def infer(self, img):
        # 预处理
        data = self.preprocess(img)
        # 预测
        result = self.predict(self.predictor, [self.infer_im_shape, data, self.scale_factor])
        return result
    
    def batch_infer(self, img_list):
        data_list = []
        for img in img_list:
            data_list.append(self.preprocess(img))
        data = np.concatenate(data_list,0)
        result = self.predict(self.predictor, [self.infer_im_shape, data, self.scale_factor])
        return result

    # ——————————————————后处理函数————————————————— #
    def draw_bbox_image(self, frame, result, threshold=0.5):
        if result != []:
            for res in result:
                cat_id, score, bbox = res[0], res[1], res[2:]
                num_id = int(cat_id)
                color = self.color_list[num_id]
                if score < threshold:
                    continue
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                label_id = self.label_list[num_id]
                cv2.rectangle(frame, (int(xmin), int(ymin-15)), (int(xmin+60), int(ymin)), color, -1)
                cv2.putText(frame, label_id+str(round(score,2)), (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        return frame

    # ————————————————————————滤波器————————————————————————　#
    def object_filter(self, result, threshold=0.5):
        object_list = []
        label_id = self.label_list[num_id]
        # 全部帧加一个0
        for i in range(len(self.label_list)):
            self.object_filter_list[i].pop(0)
            self.object_filter_list[i].append(0)
        # 逐个判断
        for res in result:
            cat_id, score, bbox = res[0], res[1], res[2:]
            num_id = int(cat_id)
            if score >= threshold:
                # 如果识别到
                self.object_filter_list[num_id][-1] = 1
                if self.object_filter_list[num_id].count(1) > self.filter_rate: # 如果出现次数大于 filter_rate ,则认为识别到
                    object_list.append(label_id)
            else:
                continue
        return object_list
    
    def object_filter_show(self, frame, result, threshold=0.5):
        object_list = []
        # 全部帧加一个0
        for i in range(len(self.label_list)):
            self.object_filter_list[i].pop(0)
            self.object_filter_list[i].append(0)
        # 逐个判断
        for res in result:
            cat_id, score, bbox = res[0], res[1], res[2:]
            num_id = int(cat_id)
            label_id = self.label_list[num_id]
            if score >= threshold:
                # 识别到次数+1
                self.object_filter_list[num_id][-1] = 1
                # 如果出现次数大于 filter_rate ,则认为识别到
                if self.object_filter_list[num_id].count(1) > self.filter_rate: 
                    object_list.append(label_id)
                    # 画图
                    color = self.color_list[num_id]
                    xmin, ymin, xmax, ymax = bbox
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                    # #cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
                    cv2.rectangle(frame, (int(xmin), int(ymin-15)), (int(xmin+60), int(ymin)), color, -1)
                    cv2.putText(frame, label_id+str(round(score,2)), (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            else:
                continue
        return frame, object_list