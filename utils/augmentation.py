import math
import random

import cv2
import numpy as np

def draw_augment(image, labels):
    h, w, c = image.shape
    x_center = w//2
    for label in labels:
        if label is not None and label > 0:
            cv2.circle(image, center=(round(x_center), round(label)), radius=3, color=(0, 14, 255), thickness=3)
    return image

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def mixup(im, labels, im2, labels2, index=0):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
#    assert isinstance(labels, list) and isinstance(labels2, list)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = labels + labels2
    #import os
    #expath ='/home/yanggang/PyCharmWorkspace/BISMNetwork/expriments/augment/mixup'
    #cv2.imwrite(os.path.join(expath, f'mixup_test_image_{index}.png'), im)
    #draw_img = draw_augment(im, labels)
    #cv2.imwrite(os.path.join(expath, f'mixup_draw_image{index}.png'), draw_img)
    return im, labels

def mosaic(imgs, labels, source_img, source_label, img_size=(1024, 64, 3), index=0):
    h, w, c = imgs[0].shape
    datatype = imgs[0].dtype
    new_image = np.zeros(img_size, dtype=datatype)
    new_labels = []
    bound1 = random.randint(int(0.333 * h), int(0.666 * h))
    bound0 = random.randint(int(0.333 * bound1), int(0.666 * bound1))
    bound2 = random.randint(bound1 + int(0.333 * (h - bound1)), bound1 + int(0.666 * (h - bound1)))
    start0, end0 = 0, bound0
    start1, end1 = bound0, bound1
    start2, end2 = bound1, bound2
    start3, end3 = bound2, h
    new_image[start0: end0, :, :] = imgs[0][start0: end0,:,  :]
    new_image[start1: end1, :, :] = imgs[1][start1: end1, :, :]
    new_image[start2: end2, :, :] = imgs[2][start2: end2, :, :]
    new_image[start3: end3, :, :] = imgs[3][start3: end3, :, :]
    new_labels.append(labels[0]) if labels[0] > start0 and labels[0] < end0 else new_labels.append(-1.0)
    new_labels.append(labels[1]) if labels[1] > start1 and labels[1] < end1 else new_labels.append(-1.0)
    new_labels.append(labels[2]) if labels[2] > start2 and labels[2] < end2 else new_labels.append(-1.0)
    new_labels.append(labels[3]) if labels[3] > start3 and labels[3] < end3 else new_labels.append(-1.0)

    if source_label > start0 and source_label < end0:
        new_image[start0: end0, :, :] = source_img[start0: end0,:,  :]
        new_labels[0] = source_label
    if source_label > start1 and source_label < end1:
        new_image[start1: end1, :, :] = source_img[start1: end1,:,  :]
        new_labels[1] = source_label
    if source_label > start2 and source_label < end2:
        new_image[start2: end2, :, :] = source_img[start2: end2,:,  :]
        new_labels[2] = source_label
    if source_label > start3 and source_label < end3:
        new_image[start3: end3, :, :] = source_img[start3: end3,:,  :]
        new_labels[3] = source_label
    #expath = '/home/yanggang/PyCharmWorkspace/BISMNetwork/expriments/augment/mosaic'
    #import os
    #cv2.imwrite(os.path.join(expath,f'mosaic_test_image_{index}.png'), new_image)
    #draw_img = draw_augment(new_image, new_labels)
    #cv2.imwrite(os.path.join(expath,f'mosaic_draw_image_{index}.png'), draw_img)

    return new_image, new_labels


