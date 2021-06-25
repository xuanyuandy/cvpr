import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import datetime


# import matplotlib
# matplotlib.use('TkAgg')
#imgtime = 1

def merge_images(img1, img2):
    # 初始化SIFT检测子
    image1 = cv.normalize(img1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image2 = cv.normalize(img2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1, None)

    kp2, des2 = sift.detectAndCompute(image2, None)

    # 检测关键点:
    img1_sift_keypoints = img1.copy()
    img2_sift_keypoints = img2.copy()
    img1_sift_keypoints = cv.drawKeypoints(img1, kp1, img1_sift_keypoints,
                                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_sift_keypoints = cv.drawKeypoints(img2, kp2, img2_sift_keypoints,
                                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.subplot(211)
    #plt.imshow(img1_sift_keypoints)
    #plt.subplot(212)
    #plt.imshow(img2_sift_keypoints)
    #plt.show()

    # BFMatcher 使用默认参数进行匹配

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用ratio检测，如果两个最相邻之间的距离之差足够大，那么就确认为是一个好的匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    point1 = []
    point2 = []
    for each in good:
        point1.append(kp1[each[0].queryIdx].pt)
        point2.append(kp2[each[0].trainIdx].pt)

    point1 = np.array(point1)
    point2 = np.array(point2)

    # 判断哪张图片在左边，如果位置相反，那么就交换两者
    num1 = 0
    num2 = 0
    flag = img1.shape[1] / 2
    for each in point1:
        if each[0] > flag:
            num1 += 1
        else:
            num2 += 1
    if num1 < num2:
        temp = img1
        img1 = img2
        img2 = temp
        temp = point1
        point1 = point2
        point2 = temp
    else:
        pass

    # 使用findHomography函数来求单应矩阵
    H, mask = cv.findHomography(point2, point1, cv.RANSAC)
    print("shape")
    print(img2.shape[0], img2.shape[1])

    print("change")
    print(H[0], H[1])

    # 计算最终拼接图片的大小
    # 均使用齐次坐标进行变换,之后
    img2_leftup = [0, 0, 1]
    img2_leftdown = [0, img2.shape[0] - 1, 1]
    img2_rightup = [img2.shape[1] - 1, 0, 1]
    img2_rightdown = [img2.shape[1] - 1, img2.shape[0] - 1, 1]
    print(img2_rightup)
    print(img2_rightdown)

    x1 = np.dot(img2_leftup, H[0])
    x2 = np.dot(img2_leftdown, H[0])
    x3 = np.dot(img2_rightup, H[0])
    x4 = np.dot(img2_rightdown, H[0])

    print("x")
    print(x1,x2,x3,x4)
    y1 = np.dot(img2_leftup, H[1])
    y2 = np.dot(img2_leftdown, H[1])
    y3 = np.dot(img2_rightup, H[1])
    y4 = np.dot(img2_rightdown, H[1])
    print("y")
    print(y1,y2,y3,y4)

    # 选择最终输出图片的尺寸
    y_out = int(max(y2, y4, img1.shape[0] - 1))
    # x_out = int(max(x3, x4) + img1.shape[0] - 1)
    x_out = int(x3) + int(x4)
    print(x_out,y_out)

    # 对右边的图片进行变换，得到变换后的图像
    img_out = cv.warpPerspective(img2, H, (x_out, y_out))
    # 获取变换后图像的所占部分
    mask = np.zeros((img2.shape[0], img2.shape[1]), np.uint8)  
    mask.fill(1)
    #print(mask.shape)
    img_out_mask = cv.warpPerspective(mask, H, (x_out, y_out))
    # img_out_mask 与 img_out 大小相同（单通道），0部分为padding，1部分为图像位置
    #global imgtime
    #cv2.imwrite('./imgmid' + str(imgtime) + '.png', img_out)
    #cv2.imwrite('./imgmask' + str(imgtime) + '.png', img_out_mask)
    #imgtime += 1
    # return img_out
    # 将变换后的图片和左边的图片拼接
    for i in range(img_out.shape[0]):
        x_temple = x1 + (x2 - x1) / (y2 - y1) * (i - y1)
        #print('xtemple:', x_temple)
        for j in range(img_out.shape[1]):
            #if j < x_temple:
            if img_out_mask[i, j] == 0: # 不在mask里，用img1
                if i < img1.shape[0] - 1 and j < img1.shape[1] - 1:
                    img_out[i, j] = img1[i, j]
                else:
                    img_out[i, j] = img_out[i, j]
            elif j > img1.shape[1] - 1:
                img_out[i, j] = img_out[i, j]
            else:
                if i < img1.shape[0] - 1:
                    img_out[i, j] = (img1.shape[1] - 1 - j) / (img1.shape[1] - 1 - x_temple) * img1[i, j] + (
                            j - x_temple) / (img1.shape[1] - 1 - x_temple) * img_out[i, j]
                else:
                    img_out[i, j] = img_out[i, j]

    return img_out


k = 4
# 读入图像
img = []
img.append(cv.imread("./image/1_resize.jpg"))
for i in range(1, k + 1):
    img.append(cv.imread("./image/" + str(i) + "_resize.jpg"))
    print(img[i])


# 将[i,j]之间的所有图像全部进行拼接
def merge(i, j):
    if (i + 1 == j):
        return merge_images(img[i], img[j])
    result = merge_images(merge(i, j - 1), img[j])
    return result


starttime = datetime.datetime.now()
img = merge(1,4)
# img2 = merge(3,5)
# img = merge_images(img1,img2)

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
cv.imwrite('./image/result.png', img)
