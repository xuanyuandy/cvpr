import os

import cv2
import numpy as np


FP = 'image/'
SP = 'temp/'

def resize(list_of_img):
    for img_num in list_of_img:
        img_path = 'temp'+str(img_num)+os.sep+'result.png'
        img = cv2.imread(img_path)
        img = cv2.resize(img,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
        img_rpath = 'temp' + str(img_num) + os.sep + 'result_resize.png'
        cv2.imwrite(img_rpath,img)
        print(img_rpath)

if __name__ == '__main__':
    list = [3,7,11]
    resize(list)
