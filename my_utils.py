import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
import numpy as np
from ensemble_boxes import *

#calculate the area of bounding boxes
def area(box_data):
  x_min = box_data[0]
  y_min = box_data[1]
  x_max = box_data[2]
  y_max = box_data[3]
  height = y_max - y_min
  width = x_max - x_min
  return height*width

#convert into submission format
def format_prediction_string(boxes, scores):
    if(boxes.size == 0):
      return 'no_box'

    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0} {1} {2} {3}".format(j[1][0], j[1][1], j[1][2], j[1][3]))

    return ";".join(pred_strings)

#for plotting purpose
def format_prediction_string_plot(boxes, scores):
    if(boxes.size == 0):
      return 'no_box'
      
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

#get bounding boxes for plotting purpose
def get_bboxes(pred_str):
    span=5
    preds = pred_str.split()
    bboxes = [list(map(int, preds[i+1:i+span] )) for i in range(0, len(preds), span)]
    return bboxes

# augmentation function for training data
def get_train_augs():
    return albumentations.Compose([
        albumentations.Flip(p=0.60),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.50),
        albumentations.HueSaturationValue(p=0.60),
        ToTensorV2()
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# augmentation function for validation data
def get_valid_augs():
    return albumentations.Compose([
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Advance augmentation function for training data
def get_train_augs_adv():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

# For filtering outputs during inference
def filter_outputs(predictions, image_index, method='soft_nms', image_size=1024, iou_thr=0.5, skip_box_thr=0.3, weights=None):
    #reference : https://github.com/ZFTurbo/Weighted-Boxes-Fusion

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    if(method == 'wbf'):
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif(method == 'soft_nms'):
        boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, method=2, iou_thr=iou_thr, thresh=skip_box_thr)
    elif(method == 'nmw'):
        boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        boxes = np.array(boxes[0])
        scores = scores[0]
        labels = labels[0]
    boxes = boxes*(image_size-1)
    return boxes, scores, labels



def make_predictions(net, images, score_threshold=0.20):
    images = torch.stack(images).float().cuda()
    predictions = []
    with torch.no_grad():
        outputs = net(images)
        for i in range(images.shape[0]):
            boxes = outputs[i]['boxes'].data.cpu().numpy()   
            scores = outputs[i]['scores'].data.cpu().numpy()
            indexes = np.where(scores > score_threshold)[0]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]