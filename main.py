import os

import cv2
import numpy as np

FP = 'image/'
SP = 'temp/'


def cvshow(name, img, dir=SP):
    # cv2.imshow(name, img)
    print(dir + name)
    cv2.imwrite(dir + name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


# 全景拼接
def siftimg_rightlignment(img_right, img_left, dir=SP):
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵

        # 将图片右进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))

        cvshow('result_medium.png', result, dir)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def merge(img_1, img_2, dir=SP, fx=0.5, fy=0.3):
    # 读取拼接图片（注意图片左右的放置）
    # 是对右边的图形做变换
    if not os.path.exists(dir):
        os.makedirs('./' + dir)
    img_right = cv2.imread(FP + img_1)
    img_left = cv2.imread(FP + img_2)

    img_right = cv2.resize(img_right, None, fx=fx, fy=fy)
    # 保证两张图一样大
    img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))

    kpimg_right, kp1, des1 = sift_kp(img_right)
    kpimg_left, kp2, des2 = sift_kp(img_left)

    # 同时显示原图和关键点检测后的图
    cvshow('img_1.png', np.hstack((img_left, kpimg_left)), dir)
    cvshow('img_2.png', np.hstack((img_right, kpimg_right)), dir)
    goodMatch = get_good_match(des1, des2)

    all_goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch, None, flags=2)

    # goodmatch_img自己设置前多少个goodMatch[:10]
    goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch[:10], None, flags=2)

    cvshow('keypoint_matches_1.png', all_goodmatch_img, dir)
    cvshow('keypoint_matches_2.png', goodmatch_img, dir)

    # 把图片拼接成全景图
    result = siftimg_rightlignment(img_right, img_left, dir)
    cvshow('result.png', result, dir)
    cvshow('result_2.png', crop(result), dir)



if __name__ == '__main__':
    merge('img (2).jpg', 'img (1).jpg', "temp2/", 0.5, 0.3)
