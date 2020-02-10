import os
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
import pickle
import sys

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def generate(img_in, img_gen):
    img_generated = img_in.numpy()
    this_path = os.path.dirname(os.path.abspath(__file__))
    file = open('{}/localdata/color_dict'.format(this_path), 'rb')
    color_dict = pickle.load(file)
    file.close()
    image_gray = rgb2gray(img_in)
    # do a filtering
    recommends = []
    recommends_gray = []
    differences = []
    for c in img_gen:    
        c_gray = rgb2gray(c)
        diff_gray = np.array(image_gray - c_gray)
        diff_gray[diff_gray<0.5]=0
        diff_gray[diff_gray>=0.5]=1
        diff_gray = diff_gray.astype("uint8")
        difference = np.sum(diff_gray)
        # only keep significant generated designs
        if difference>100:
            recommends.append(c)
            recommends_gray.append(diff_gray)
            differences.append(difference)
    try:
        top_recommend = [recommends[ind] for ind in np.argpartition(differences, -2)[-2:]]
        top_recommend_gray = [recommends_gray[ind] for ind in np.argpartition(differences, -2)[-2:]]
        for i in range(len(top_recommend)):
            img = top_recommend[i].numpy()
            cnts = cv2.findContours(top_recommend_gray[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            img_add = np.ones_like(img)
            for c in cnts:
                mask = np.zeros(img.shape[:2], dtype="uint8")
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(mask,(x,y),(x+w,y+h),1,-1)
                mask = cv2.erode(mask, None, iterations=3)

                if (np.max(mask)==1):
                    mean = cv2.mean(img, mask=mask)[:3]
                    minDist = np.inf
                    for label,color in color_dict.items():
                        d = dist.euclidean(color, mean)
                        if d < minDist:
                            minDist = d
                            final_label = label
                            final_color = color
                    img_add= cv2.rectangle(img_add,(x,y),(x+w,y+h),final_color,-1)
            img_generated = np.minimum(img_in, img_add)
            return img_generated
    except:
        return img_generated