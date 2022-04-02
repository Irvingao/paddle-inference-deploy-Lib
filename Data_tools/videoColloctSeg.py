import cv2
import time
import os

def check_video_dir(save_dir):
    if os.path.isdir(save_dir):
        folder = os.listdir(save_dir)
        print(folder)
        video_id = 0
        if folder != []:
            video_id_list =[]
            for i in folder:
                video_id = int(i.split(".")[0])
                video_id_list.append(video_id)
            video_id = max(video_id_list)
            print("There are already {} videos saved.".format(video_id+1))
            return video_id+1
    else:
        os.mkdir(save_dir)
        print("There is no video saved.")
        return 0
            

def save_video(cap, video_id, save_dir):

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(save_dir, '{}.avi'.format(video_id)),fourcc, 30.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,1)
            out.write(frame) # 保存视频
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__': 

    save_dir = "Dataset_segmentation"
    video_id = check_video_dir(save_dir)

    cap = cv2.VideoCapture(0)

    save_video(cap, video_id, save_dir)
    
