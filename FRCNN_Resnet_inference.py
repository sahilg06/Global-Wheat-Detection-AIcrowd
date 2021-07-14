import os
import sys
import numpy as np 
import pandas as pd

import cv2

import torch
from PIL import Image

from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import models
import odach as oda # Test time augmentation(TTA)
import my_utils
from datasets import WheatDataset_test

#base_dir = "/raid/sahil_g_ma/wheatDetection"
base_dir = '/workspace/wheatDetection'

sys.path.append(os.path.join(base_dir, 'detection'))
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# boxes with scores > Detection_threshold are considered
DETECTION_THRESHOLD = 0.30

# sample submission
sample_sub_df = pd.read_csv(os.path.join(base_dir, 'submissions', 'submission.csv'))

model = models.FRCNN_resnetfpn_backbone('resnet152', pre_trained=False)
model.to(DEVICE)

# loading the trained model
model.load_state_dict(torch.load(os.path.join(base_dir, 'saved_models', 'frcnn_resnet152fpn_ignore_nobox5_pseudo2.pth'), 
                                 map_location=DEVICE))

# evaluation mode
model.eval()

test_dataset = WheatDataset_test(sample_sub_df, base_dir)

test_data_loader = DataLoader(
    test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
    drop_last = False,
    collate_fn = utils.collate_fn
)

######################################## For Test Time augmentation
# tta = [oda.HorizontalFlip(), oda.VerticalFlip(), oda.Rotate90(), oda.Multiply(0.9), oda.Multiply(1.1)]
# # wrap model and tta
# tta_model = oda.TTAWrapper(model, tta, skip_box_thr=0.3)

# # For TTA
# idx = 0
# results = []
# print("Starting Inference......")
# for image_ids, images, domains in test_data_loader:
#     #images = list(image.to(DEVICE) for image in images)
#     images = images.to(DEVICE)
#     outputs = model(images)
#     boxes_ = outputs[0]['boxes'].data.cpu().numpy()
#     if((idx+1) % 1000 == 0):
#          print(f'{idx+1} batches done')
#     idx += 1
#     if(boxes_.size!=0):
#         boxes, scores, labels = tta_model(images) #implementing tta
#         boxes = boxes[scores > DETECTION_THRESHOLD]
#         scores = scores[scores > DETECTION_THRESHOLD]
#         boxes = boxes*1024
#         boxes = boxes.astype(np.int32).clip(min=0, max=1023)
#     else:
#         boxes = boxes_
#     result = {
#         'image_name': image_ids,
#         'domain' : domains,
#         'PredString': my_utils.format_prediction_string(boxes, scores)
#     }
#     results.append(result)
# print("All batches done")



############################################# For filtering outputs 
# idx = 0
# results = []
# print("Starting Inference......")
# for image_ids, images, domains in test_data_loader:
#     images = list(image.to(DEVICE) for image in images)
#     if((idx+1) % 100 == 0):
#          print(f'{idx+1} batches done')
#     predictions = my_utils.make_predictions(model, images)
#     idx += 1
#     for i, image in enumerate(images):
#         boxes, scores, labels = my_utils.filter_outputs(predictions, image_index=i, method='wbf', iou_thr=0.35, skip_box_thr=0.3)
#         boxes = boxes.astype(np.int32)
#         image_id = image_ids[i]
#         domain = domains[i]
#         result = {
#             'image_name': image_id,
#             'domain' : domain,
#             'PredString': my_utils.format_prediction_string(boxes, scores)
#         }
#         results.append(result)
# print("All batches done")



############################################## Normal Inference
idx = 0
results = []
results_polt = []
print("Starting Inference......")
for image_ids, images, domains in test_data_loader:
    if((idx+1) % 100 == 0):
        print(f'{idx+1} batches done')
    images = list(image.to(DEVICE) for image in images)
    outputs = model(images)
    idx += 1
    for i, image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
        scores = scores[scores >= DETECTION_THRESHOLD]
        image_id = image_ids[i]
        domain = domains[i]
        result = {
            'image_name': image_id,
            'domain' : domain,
            'PredString': my_utils.format_prediction_string(boxes, scores)
        }
        result_plot = {
            'image_name': image_id,
            'PredString': my_utils.format_prediction_string_plot(boxes, scores)
        }
        results.append(result)
        results_polt.append(result_plot)
print("All batches done")


# final submission
sub_df = pd.DataFrame(results)
#sub_df_plot = pd.DataFrame(results_polt)


print("Saving your file")
sub_df.to_csv('final_sub.csv', index=False)
print("File saved")



# Plotting some resluts
# for image_id, pred_str in zip(sub_df_plot.iloc[:3]['image_name'], sub_df_plot.iloc[:3]['PredString']):
#     image_path = os.path.join(base_dir, 'test', f'{image_id}.png')
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     image /= 255.0
#     boxes = my_utils.get_bboxes(pred_str)

#     fig, ax = plt.subplots(1, 1, figsize=(16, 8))

#     for box in boxes:
#         cv2.rectangle(image,
#                       (box[0], box[1]),
#                       (box[2], box[3]),
#                       (255, 0, 0), 3)

#     ax.set_axis_off()
#     ax.imshow(image)
#     plt.show(block = True)
