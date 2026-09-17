# Copyright CNRS 2026

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.engine.results import Results
import yolov5
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

import warnings

DFYOLO_NAME = "DF"
DFYOLO_WIDTH = 960 # image width
DFYOLO_THRES = 0.6
DFPATH = os.path.abspath(os.path.dirname(__file__))
DFYOLO_WEIGHTS = os.path.join(DFPATH, 'deepfaune-yolov8s_960.pt')

MDSYOLO_NAME = "MDS"
MDSYOLO_WIDTH = 960 # image width
MDSYOLO_THRES = 0.5
MDSYOLO_WEIGHTS = os.path.join(DFPATH, 'md_v1000.0.0-sorrel.pt')

MDRYOLO_NAME = "MDR"
MDRYOLO_WIDTH = 1280
MDRYOLO_THRES = 0.5
MDRYOLO_WEIGHTS = os.path.join(DFPATH, 'md_v1000.0.0-redwood.pt')

class YOLOEnsemble:
    def __init__(self, weightA, weightB=None, imgszA=None, imgszB=None, thresA=None, thresB=None, backstop=True, device=None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.yoloA = YOLO(weightA).to(device)
        self.yoloB = None if weightB is None else YOLO(weightB).to(device)
        self.imgszA = imgszA
        self.imgszB = imgszB
        self.thresA = thresA
        self.thresB = thresB
        self.backstop = backstop

    def __call__(self, filename_or_imagecv=None, verbose=False, device=None):
        try:
            resultsA = self.yoloA(filename_or_imagecv, verbose=verbose, imgsz=self.imgszA, conf=self.thresA)
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as err:
            print(err)
            raise err
        # Single model case:
        if self.yoloB is None:
            return resultsA        
        # Two models case:
        # Are there any relevant boxes found by A?
        detectionA = resultsA[0].cpu().numpy().boxes
        # Yes. Stop here in backstop mode or continue in ensemble mode
        if len(detectionA.cls) > 0 and self.backstop:
            return resultsA
        # Are there any relevant boxes found by B?
        resultsB = self.yoloB(filename_or_imagecv, verbose=verbose, imgsz=self.imgszB, conf=self.thresB)
        detectionB = resultsB[0].cpu().numpy().boxes
        # Yes. Stop here in backstop mode
        if self.backstop:
            return resultsB        
        # Concat A and B in ensemble mode
        boxes = np.concatenate((detectionA.xyxy, detectionB.xyxy))
        scores = np.concatenate((detectionA.conf, detectionB.conf))
        classes = np.concatenate((detectionA.cls, detectionB.cls))
        # NMS per category
        keep = list(batched_nms(torch.Tensor(boxes), torch.Tensor(scores), torch.Tensor(classes), iou_threshold=0.7).numpy())        
        resultsA[0].update(np.concatenate((boxes, np.expand_dims(scores, 1), np.expand_dims(classes, 1)), axis=1)[keep]) # update is replace
        return resultsA

class MDRedwood:
    IMAGE_SIZE = MDRYOLO_WIDTH  # The class must have an IMAGE_WIDTH attribute

    def __init__(self, weight=None, imgsz=None, thres=None, device=None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self.thres = thres
        checkpoint = torch.load(MDRYOLO_WEIGHTS, map_location=device, weights_only=False)
        self.model = checkpoint["model"].float().fuse().eval().to(self.device)
        for m in self.model.modules():
            if isinstance(m, torch.nn.Upsample):
                m.recompute_scale_factor = None
        
    def transform(self, np_img):
        img = letterbox(np_img, new_shape=self.imgsz, stride=64, auto=False)[0]
        img = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).float() / 255.0
        return img
    
    def __call__(self, filename_or_imagecv=None, verbose=False, device=None):
        try:
            img = cv2.imread(filename_or_imagecv) if isinstance(filename_or_imagecv, str) else filename_or_imagecv
            imagecv = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            preds = self.model(imagecv.unsqueeze(0).to(self.device))[0]
            preds = torch.cat(non_max_suppression(prediction=preds, conf_thres=self.thres), axis=0)
            preds[:, :4] = scale_boxes([self.IMAGE_SIZE] * 2, preds[:, :4], img.shape).round()
            results = [Results(orig_img=img, path="", names={0: "animal", 1: "person", 2: "vehicle"})]
            results[0].update(preds)
            return results
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as err:
            print(err)
            raise err


####################################################################################
### BEST BOX DETECTION 
####################################################################################
class Detector:
    def __init__(self, name=DFYOLO_NAME, device=None):
        print(f"Using {name} for detection")
        self.device = device
        if name not in [DFYOLO_NAME, MDSYOLO_NAME, DFYOLO_NAME+"bs"+MDSYOLO_NAME, DFYOLO_NAME+MDSYOLO_NAME,
                        MDRYOLO_NAME]:
            name = DFYOLO_NAME
            warnings.warn("Detector model "+name+" not found. Using "+DFYOLO_NAME+" instead.")
        if name == DFYOLO_NAME:
            self.yolo = YOLOEnsemble(DFYOLO_WEIGHTS, imgszA=DFYOLO_WIDTH, thresA=DFYOLO_THRES, device=device)
        if name == MDSYOLO_NAME:
            self.yolo = YOLOEnsemble(MDSYOLO_WEIGHTS, imgszA=MDSYOLO_WIDTH, thresA=MDSYOLO_THRES, device=device)
        if name == DFYOLO_NAME+"bs"+MDSYOLO_NAME: # backstop method
            self.yolo = YOLOEnsemble(DFYOLO_WEIGHTS, MDSYOLO_WEIGHTS, imgszA=DFYOLO_WIDTH, imgszB=MDSYOLO_WIDTH,
                                     thresA=DFYOLO_THRES, thresB=MDSYOLO_THRES, backstop=True, device=device)
        if name == DFYOLO_NAME+MDSYOLO_NAME: # ensemble method
            self.yolo = YOLOEnsemble(DFYOLO_WEIGHTS, MDSYOLO_WEIGHTS, imgszA=DFYOLO_WIDTH, imgszB=MDSYOLO_WIDTH,
                                     thresA=DFYOLO_THRES, thresB=MDSYOLO_THRES, backstop=False, device=device)
        if name == MDRYOLO_NAME:
            self.yolo = MDRedwood(MDRYOLO_WEIGHTS, MDRYOLO_WIDTH, MDRYOLO_THRES, device=device)

    def bestBoxDetection(self, filename_or_imagecv):
        try:
            results = self.yolo(filename_or_imagecv, device=self.device)
        except FileNotFoundError:
            return None, 0, np.zeros(4), 0, []
        except Exception as err:
            print(err)
            return None, 0, np.zeros(4), 0, []
        # orig_img a numpy array (cv2) in BGR
        imagecv = results[0].cpu().orig_img
        detection = results[0].cpu().numpy().boxes

        # Are there any relevant boxes?
        if not len(detection.cls):
            return None, 0, np.zeros(4), 0, []
        else:
            # Yes. Non empty image
            pass
        # Is there a relevant animal box? 
        try:
            # Yes. Selecting the best animal box
            kbox = np.where(detection.cls==0)[0][0]
        except IndexError:
            # No: Selecting the best box for another category (human, vehicle)
            kbox = 0
        # categories are 1=animal, 2=person, 3=vehicle and the empty category 0=empty
        category = int(detection.cls[kbox]) + 1
        box = detection.xyxy[kbox] # xmin, ymin, xmax, ymax
        # Is this an animal box ?
        if category == 1:
            # Yes: cropped image is required for classification
            croppedimage = cropSquareCVtoPIL(imagecv, box.copy())
        else: 
            # No: cropped image is not required for classification 
            croppedimage = None
        ## animal count
        if category == 1:
            count = sum(detection.cls==0) # only above a threshold
        else:
            count = 0
        ## human boxes
        ishuman = (detection.cls==1)
        if any(ishuman==True):
            humanboxes = detection.xyxy[ishuman,]
        else:
            humanboxes = []
        return croppedimage, category, box, count, humanboxes

    def merge(self, detector):
        pass
    
####################################################################################
### BEST BOX DETECTION WITH JSON
####################################################################################
from load_api_results import load_api_results
import json
import contextlib
import os
from pandas import concat
from numpy import argmax
import sys

MDV5_THRES = 0.4

class DetectorJSON:
    """
    We assume JSON categories are 1=animal, 2=person, 3=vehicle and the empty category 0=empty

    :param jsonfilename: JSON file containing the bondoing boxes coordinates, such as generated by megadetectorv5
    """
    def __init__(self, jsonfilename, thres=MDV5_THRES):
        # getting results in a dataframe
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            try:
                self.df_json, _ = load_api_results(jsonfilename)
                # removing lines with Failure event
                if 'failure' in self.df_json.keys():
                    self.df_json = self.df_json[self.df_json['failure'].isnull()]
                    self.df_json.reset_index(drop=True, inplace = True)
                    self.df_json.drop('failure', axis=1, inplace=True)
            except json.decoder.JSONDecodeError:
                self.df_json = []
        self.thres = thres
        self.k = 0 # current image index
        self.kbox = 0 # current box index
        self.imagecv = None
        self.filenameindex = dict()
        self.setFilenameIndex()

    def bestBoxDetection(self, filename):
        try:
            self.k = self.filenameindex[filename]
        except KeyError:
            return None, 0, np.zeros(4), 0, []
        # now reading filename to obtain width/height (required by convertJSONboxToBox)
        # and possibly crop if it is an animal
        self.nextImread() 
        if len(self.df_json['detections'][self.k]): # is non empty
            # Focus on the most confident bounding box coordinates
            self.kbox = argmax([box['conf'] for box in self.df_json['detections'][self.k]])
            if self.df_json['detections'][self.k][self.kbox]['conf']>self.thres:
                category = int(self.df_json['detections'][self.k][self.kbox]['category'])
            else:
                category = 0 # considered as empty
            count = sum([box['conf']>self.thres for box in self.df_json['detections'][self.k]])
        else: # is empty
            category = 0
        if category == 0:
            return None, 0, np.zeros(4), 0, []
        # is an animal detected ?
        if category != 1:
            croppedimage = None
            box = self.convertJSONboxToBox()
        # if yes, cropping the bounding box
        else:
            croppedimage, box = self.cropCurrentBox()
            if croppedimage is None: # FileNotFoundError
                category = 0
        ## human boxes for compatbility, not supported here
        humanboxes = []
        return croppedimage, category, box, count, humanboxes

    def nextBoxDetection(self):
        if self.k >= len(self.df_json):
            raise IndexError # no next box
        # is an animal detected ?
        if len(self.df_json['detections'][self.k]):
            if self.kbox == 0:
                self.nextImread() 
            # is box above threshold ?
            if self.df_json['detections'][self.k][self.kbox]['conf']>self.thres:
                category = int(self.df_json['detections'][self.k][self.kbox]['category'])
                croppedimage, box = self.cropCurrentBox()
            else: # considered as empty
                category = 0
                croppedimage = None
            self.kbox += 1
            if self.kbox >= len(self.df_json['detections'][self.k]):
                self.k += 1
                self.kbox = 0
        else: # is empty
            category = 0
            croppedimage = None
            self.k += 1
            self.kbox = 0
        return croppedimage, category

    def convertJSONboxToBox(self):
        box_norm = self.df_json['detections'][self.k][self.kbox]["bbox"]
        height, width, _ = self.imagecv.shape 
        xmin = int(box_norm[0] * width)
        ymin = int(box_norm[1] * height)
        xmax = xmin + int(box_norm[2] * width)
        ymax = ymin + int(box_norm[3] * height)
        box = [xmin, ymin, xmax, ymax]
        return(box)
        
    def cropCurrentBox(self):
        if self.imagecv is None:
            return None, np.zeros(4)
        box = self.convertJSONboxToBox()
        croppedimage = cropSquareCVtoPIL(self.imagecv, box)
        return croppedimage, box
    
    def setFilenameIndex(self):
        k = 0
        for filename in self.getFilenames():
            self.filenameindex[filename] = k
            k = k+1
        
    def getNbFiles(self):
        return self.df_json.shape[0]
    
    def getFilenames(self):
        return list(self.df_json["file"].to_numpy())
    
    def getCurrentFilename(self):
        if self.k >= len(self.df_json):
            raise IndexError
        return self.df_json['file'][self.k]
    
    def nextImread(self):
        try:
            self.imagecv = cv2.imdecode(np.fromfile(str(self.df_json["file"][self.k]), dtype=np.uint8),  cv2.IMREAD_UNCHANGED)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            self.imagecv = None

    def resetDetection(self):
        self.k = 0
        self.kbox = 0
    
    def merge(self, detector):
        self.df_json = concat([self.df_json, detector.df_json], ignore_index=True)
        self.resetDetection()
        self.setFilenameIndex()


  
####################################################################################
### TOOLS
####################################################################################      
'''
:return: cropped PIL image, as squared as possible (rectangle if close to the borders)
'''
def cropSquareCVtoPIL(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0,int(y1)):min(int(y2),height),max(0,int(x1)):min(int(x2),width)]
    croppedimage = Image.fromarray(croppedimagecv[:,:,(2,1,0)]) # converted to PIL BGR image
    return croppedimage
