import cv2
import os
import re

IMG_PATH = './out'
VID_PATH = './tescik.avi'


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_video(imgpath=IMG_PATH):

    images = [img for img in os.listdir(imgpath) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(imgpath, images[0]))
    height, width, layers = frame.shape
    images.sort(key=natural_keys)
    print(images)

    video = cv2.VideoWriter(VID_PATH, 0, 8, (width, height))

    print("Creating the video")
    for image in images:
        video.write(cv2.imread(os.path.join(imgpath, image)))

    cv2.destroyAllWindows()
    video.release()
    print("Video Created")


create_video()

