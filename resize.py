import glob
import os

from PIL import Image


def ResizeImage(filein, fileout, width=1480, height=666, type='jpeg'):
    img = Image.open(filein)
    out = img.resize((width, height), Image.ANTIALIAS)
    # resize image with high-quality
    out.save(fileout, type)


if __name__ == '__main__':
    width = 480
    height = 1056
    images = glob.glob('/Users/dy/PycharmProjects/cv_exp/sift/image' + os.sep + '**.jpg')
    for img in images:
        file_out=os.path.splitext(img)[0] + '_resize.jpg'
        ResizeImage(img,file_out)
