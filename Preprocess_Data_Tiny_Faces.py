
#In this code, we take the images and their corresponding XML file, if exists. 
#Extract bounding boxes from the XML file for the malignant cases, if not found add 0 to the detections. 
#Write all the detections for all the images in a single file. 
#The another output is to resize all images to 1000x1000 and save it in new directory.
#Note: No data augmentation is done here.

import numpy as np
from PIL import Image
from lxml import etree
import os
import cv2
import csv
from skimage import exposure

import matplotlib.pyplot as plt
#%matplotlib inline



DATA_DIR = "/home/sumanyu/scratch/MIA_XAI/DDSM_processed/DDSM_with_XML_Fin/"
NEW_DATA_DIR = "/home/sumanyu/scratch/MIA_XAI/DDSM_Wider_Face/images/"
OUTPUT_FILE = '/home/sumanyu/scratch/MIA_XAI/DDSM_Wider_Face/ddsm_bbox.txt'

files = os.listdir(DATA_DIR)
img_files = []
for file in files:
    if '.jpg' in file:
        img_files.append(file)
        

print(f"total files:{len(files)}")
print(f"images: {len(img_files)}")

def coord_w_h(xml_path, img_size):
    '''
    Input: xml_path: path to the xml file
    img_size: image shape
    
    This function resize the image to 1000x1000 and hence
    the corresponding bounding boxes.
    And get the bouding boxes as (x1,y1,w,h)
    
    Output: '''
    root = etree.parse(xml_path)
    root = root.getroot()
    w = img_size[1]
    h = img_size[0]
    coord = []
    names = []
    for obj in root.iter('object'):
        bbx = obj.find('bndbox')
        name = obj.find('name').text 
        size = root.find('size')
        width = int( size.find('width').text )
        height = int( size.find('height').text )
        x_scale = (1000*1.0)/width
        y_scale = (1000*1.0)/height
        
        x1 = int( bbx.find('xmin').text )
        y1 = int( bbx.find('ymin').text )
        x2 = int( bbx.find('xmax').text )
        y2 = int( bbx.find('ymax').text )
        (origLeft, origTop, origRight, origBottom) = (x1, y1, x2, y2)
        
        x1 = int(np.round(origLeft * x_scale))
        y1 = int(np.round(origTop * y_scale))
        x2 = int(np.round(origRight * x_scale))
        y2 = int(np.round(origBottom * y_scale))
        
        # convert to x1,y1,w,h format
        w = x2 - x1
        h = y2 - y1
        coord.append( [x1,y1,w,h] )
        names.append(name)
        
    if len(coord) == 0:
        x1 = 0
        w = 0
        y1 = 0
        h = 0
        coord.append( [x1,y1,x2,y2] )
        names.append("NoObject")
    return names, coord, img_size[1], img_size[0]

xml_path = DATA_DIR
img_path = DATA_DIR
save_path_xml = NEW_DATA_DIR
save_path_img = NEW_DATA_DIR

xml_cnt = 0
no_xml_cnt = 0
genuine_cnt = 0
no_mass_cnt = 0

with open(OUTPUT_FILE, "w") as writer:
    for file in img_files:
        #print(file)
        xml_old = xml_path + os.path.splitext(file)[0] + '.xml'    
        xml_new = save_path_xml + os.path.splitext(file)[0] + '.xml'
        img_old = img_path + file
        img_new = save_path_img + file   

        # GK: 0 represents the grey scale image. Internally in the code, it is changed to RGB
        # This will load the image as (height, width)
        img = cv2.imread(img_old,0)
        #print("img.shape",img.shape)
        if (img.shape == (4000,4000)):
            resize_img = img

        else:
            resize_img = cv2.resize(img, (4000,4000), interpolation = cv2.INTER_AREA)

        cv2.imwrite(img_new, resize_img)
        
        try:
            names, coord, width, height = coord_w_h(xml_old, resize_img.shape) 
            #print(names, coord, width, height)
            xml_cnt += 1
            temp_coord = []
            # there was an error as only the first detection is considered
            for i in range(len(names)):
                if (names[i] not in ['0', '2', '3', '4a', 'BENIGN', 'NoObject', 'BENIGN_WITHOUT_CALLBACK', 'UNPROVEN']):
                    # change the code to ignore these detections
                    temp_coord.append(coord[i])
        except:
            #print("No xml file")
            no_xml_cnt +=1
            temp_coord = []
            
        # write to the output txt file
        file_name = os.path.basename(img_new)
        file_name = os.path.join('ddsm_images', file_name)
        
        n = 0
        bboxes = []
        
        # for no xml file or all detections are benign
        if len(temp_coord) == 0:
            # don't increment "n" here
            bboxes.append("0 0 0 0 0 0 0 0 0 0 ")
            no_mass_cnt += 1
        else:
            # this is for masses detections
            #print("Genuine malignant detections",temp_coord)
            genuine_cnt += 1
            for bbox in temp_coord:
                # ignore invalid detections
                if bbox[2] == 0 | bbox[3] == 0:
                    #print("xml file exists but invalid detections:", file_name)
                    continue
                else:
                    bbox_str = ""
                    n += 1
                    for b in bbox:
                        bbox_str = bbox_str + str(b) + " "
                    bbox_str = bbox_str + "0 0 0 0 0 0 "
                    bboxes.append(bbox_str)
            # this is the case when all malignant detections have satisfy: bbox[2] == 0 | bbox[3] == 0
            if n == 0:
                print("When all malignant detections are invalid", file_name)
                bboxes.append("0 0 0 0 0 0 0 0 0 0 ")
                no_mass_cnt += 1
                
        # write to the output file
        writer.write(file_name + "\n")
        writer.write(str(n) + "\n")

        for bbox in bboxes:
            writer.write(bbox + "\n")
        
print(f"xml available:{xml_cnt}")
print(f"no xml: {no_xml_cnt}")
print(f"genuine_cnt: {genuine_cnt}")
print(f"no_mass_cnt: {no_mass_cnt}")

