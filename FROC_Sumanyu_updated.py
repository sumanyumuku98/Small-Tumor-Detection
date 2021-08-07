#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:43:26 2021

@author: Rohan Raju Dhanakshirur 
"""

import numpy as np
import os
import matplotlib.pyplot as plt

################################################################################
"""Please Put the path for the predicted folder here"""
path_test = 'final_files/best_model/detection_results/'
################################################################################

P_test = os.listdir(path_test)
num_test = len(P_test)
#proba_map = np.zeros([num_test,5])
proba_map = []
for i_test in range(num_test):
    f2 = open(path_test+P_test[i_test])
    l_test = f2.readlines()
    pm = np.zeros([len(l_test),5])
    for j_test in range(len(l_test)):
        words_test = l_test[j_test].split() 
        x1 = int(float(words_test[2]))
        y1 = int(float(words_test[3]))
        x2 = int(float(words_test[4]))
        y2 = int(float(words_test[5]))
        pm[j_test,:] =[x1,y1,x2,y2,float(words_test[1])]
    proba_map.append(pm)
f2.close()

################################################################################
"""Please put the path for the predicted folder for the negative data here"""
path_test_neg = 'final_files/best_model/detection_neg_test/'
################################################################################

P_test_neg = os.listdir(path_test_neg)
num_test = len(P_test_neg)
#proba_map = np.zeros([num_test,5])
for i_test in range(num_test):
    f3 = open(path_test_neg+P_test_neg[i_test])
    l_test_neg = f3.readlines()
    pm = np.zeros([len(l_test_neg),5])
    for j_test in range(len(l_test_neg)):
        words_test = l_test_neg[j_test].split() 
        x1 = int(float(words_test[2]))
        y1 = int(float(words_test[3]))
        x2 = int(float(words_test[4]))
        y2 = int(float(words_test[5]))
        pm[j_test,:] =[x1,y1,x2,y2,float(words_test[1])]
    proba_map.append(pm)
f3.close()

print("predicted_values_obtained")


###############################################################################
""" Please put the path for the ground truth bounding boxes here!"""
path = 'predicted_outputs_for_test_class/test_gnd_truth/'
################################################################################

P = os.listdir(path_test)
num = len(P)
ground_truth = []
for i in range(num):
    f1 = open(path+P[i])
    l = f1.readlines()
    gt = np.zeros([len(l),4])
    for j in range(len(l)):
        words = l[j].split() 
        x1 = int(float(words[1]))
        y1 = int(float(words[2]))
        x2 = int(float(words[3]))
        y2 = int(float(words[4]))
        gt[j,:] = [x1,y1,x2,y2]
    ground_truth.append(gt)
f1.close()

l_g = len(ground_truth)
for i in range(l_g,len(proba_map)):
    ground_truth.append([0,0,0,0])


print("ground_truth_obtained")


def find_center(bbox):
    cx = 1.0*(bbox[0] + bbox[2]) / 2
    cy = 1.0*(bbox[1] + bbox[3]) / 2
    return cx,cy

def check_true_positive(cx,cy, gt_bbox):
    for gt in gt_bbox:
        if(len(gt)==4):
            x1, y1, x2, y2 = gt
        else:
            x1, y1, x2, y2, s = gt
        #print(gt)
        cx1,cx2 = find_center([x1,y1,x2,y2])
#        if abs(cx1-cx)<5 and abs(cx2-cy)<5:
        if (cx >= x1 and cx <= x2 and cy >= y1 and cy<= y2):
            #print(score)
            #print(thres)
            #print('true')
            return True

    return False

def check_false_negative(centers, gt_bbox):
    for center in centers:
        x1, y1, x2, y2 = gt_bbox
        cx, cy = center
        #print(gt)
        if (cx >= x1 and cx <= x2 and cy >= y1 and cy<= y2):
            return False

    return True


threshold = list(np.arange(0.0, 1.001, 0.01))
th_len = len(threshold)
# threshold = [0.0]
prec_list = []
sensitivity = []
false_pos_rate = []
avg_false_pos_per_image = []
f1_list = []
th_counter = 1
for thresh in threshold:
    tp_count = 0
    fp_count = 0
    fn_count = 0
    tn_count = 0
    count=0
    centers = []
    for file in range(len(ground_truth)):
        count += 1
        pred_bbox = proba_map[file]
        gt_bbox = ground_truth[file]
        gt_bbox_bool = np.array(gt_bbox)==0
        # print(pred_bbox)
        # print(gt_bbox)
        if (len(pred_bbox)!=0):
            for j,bbox in enumerate(pred_bbox):
                if(bbox[-1]>thresh):
                    cx,cy = find_center(bbox)
                    centers.append([cx, cy])
                    
                    if(len(gt_bbox)>0) and not(gt_bbox_bool.all()):
                        if (check_true_positive(cx, cy, gt_bbox)):
                            tp_count+=1
                        else:
                            fp_count+=1
                    else:
                        #print(fp_count)
                        fp_count+=1
        else:
            if(len(gt_bbox)!=0) and not(gt_bbox_bool.all()):
                fn_count+=len(gt_bbox)
            else:
                tn_count+=1

        if(len(gt_bbox)!=0) and not(gt_bbox_bool.all()):
            for j, bbox in enumerate(gt_bbox):
                if(len(gt_bbox)>0):
                    if (check_false_negative(centers, bbox)):
                        fn_count+=1
        # print(centers)


#    print("true positive",tp_count)
#    print("false positive",fp_count)
#    print("false negative",fn_count)
#    print("true negative", tn_count)

    try:
        s = tp_count*1.0/(tp_count+fn_count)
    except:
        s=0
#    print("recall/sensitivity",s)

    a_fp = fp_count*1.0/count
#    print("Avg_fp", a_fp)
    try:
        prec = tp_count*1.0/(tp_count+fp_count)
    except:
        prec = 0
#    print('precision', prec)
    prec_list.append(prec)
    try:
        f1 = 2*prec*s*1.0/(prec+s)
    except:
        f1 = 0
#    print('f1', f1)
    f1_list.append(f1)
    try:
        fpr = fp_count*1.0/(tn_count + fp_count)
    except:
        fpr = 0
    #print("False Positive rate", fpr)

    sensitivity.append(s)
    avg_false_pos_per_image.append(a_fp)
    false_pos_rate.append(fpr)
    print('completed ',th_counter,' out of ',th_len," Length of centers is ",len(centers))
    th_counter = th_counter+1
# print(sensitivity)
# print(avg_false_pos_per_image)
# print(false_pos_rate)
for ctr in range(len(threshold)):
    print(threshold[ctr], sensitivity[ctr], avg_false_pos_per_image[ctr], false_pos_rate[ctr])
#plt.plot(avg_false_pos_per_image,sensitivity,marker='o')
#plt.plot(false_pos_rate[-900:],sensitivity[-900:],marker='x', label='ROC')
## plt.xlabel('')
#plt.title('FROC Curve')
#plt.legend()
#plt.ylabel('Sensitivity')
#plt.xlabel('Average_False_positive_Per_image')
#plt.show()

plt.plot(np.array(avg_false_pos_per_image),np.array(sensitivity),marker='o')
#plt.plot(false_pos_rate[-900:],sensitivity[-900:],marker='x', label='ROC')
# plt.xlabel('')
plt.title('FROC Curve')
plt.legend()
plt.ylabel('Sensitivity')
plt.xlabel('Average_False_positive_Per_image')
plt.show()

for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.025:
        print('Sensitivity at 0.025 FPI = ',sensitivity[i])
        break
for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.05:
        print('Sensitivity at 0.05 FPI = ',sensitivity[i])
        break
    
for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.075:
        print('Sensitivity at 0.075 FPI = ',sensitivity[i])
        break
    
for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.1:
        print('Sensitivity at 0.1 FPI = ',sensitivity[i])
        break
    
for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.15:
        print('Sensitivity at 0.15 FPI = ',sensitivity[i])
        break

for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.2:
        print('Sensitivity at 0.2 FPI = ',sensitivity[i])
        break

for i in range(len(avg_false_pos_per_image)):
    if avg_false_pos_per_image[i]<0.3:
        print('Sensitivity at 0.3 FPI = ',sensitivity[i])
        break
    
    
#np.savetxt('fpr_best.txt',avg_false_pos_per_image)
#np.savetxt('tpr_best.txt',sensitivity)