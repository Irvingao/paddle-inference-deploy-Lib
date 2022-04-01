'''
Descripttion: 
version: 
Author: Irving.Gao
Date: 2022-01-11 16:44:53
LastEditors: Irving.Gao
LastEditTime: 2022-01-11 16:47:09
'''
import cv2
from Lib.Camera_Lib import *
import time
import os

init_img_id = 0
n = 0
dataset_dir = "../data_det"

def check_img_dir(dataset_dir):
    global img_id
    if os.path.isdir(dataset_dir):
        folder = os.listdir(dataset_dir)
        print(folder)
        img_id = 0
        if folder != []:
            img_id_list =[]
            for i in folder:
                img_id = int(i.split(".")[0])
                img_id_list.append(img_id)
            img_id = max(img_id_list)
            print(img_id)
    else:
        os.mkdir(dataset_dir)
        img_id = 0
            
def save_img(img, save_dir):
    global img_id ,n 
    if n%10 == 0: # 
        cv2.imwrite(os.path.join(save_dir, str(img_id) + '.jpg'), img)
        print("total image:" + str(img_id))
        img_id+=1
    n=n+1

if __name__ == '__main__': 

    cap = Realsense()
    
    check_img_dir(dataset_dir)
    while(1):
        start = time.time()
        cap.wait_frames()
        image = cap.rgb_image()

        save_img(image, dataset_dir)

        cv2.imshow("realsense", image)
        
        k=cv2.waitKey(1)
        if k==ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
