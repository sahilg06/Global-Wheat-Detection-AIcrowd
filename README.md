# Global-Wheat-Detection-2021
This project includes a solution to [Global-Wheat-Challange-2021](https://www.aicrowd.com/challenges/global-wheat-challenge-2021).<br>
Also, It's a good source to get started with implementation of Object-Detection in **PyTorch**.

### Task
The task is to localize the wheat head contained in each image.

![wheat_data](https://user-images.githubusercontent.com/59660566/125678202-547f128a-eb5a-4e56-9890-2660247542a6.png)

### Dataset
The training dataset will be the images acquired in Europe and Canada, which cover approximately 4000 images and the test dataset will be composed of the images from North America (except Canada), Asia, Oceania and Africa and covers approximately 2000 images.
You can download the dataset from [here](https://www.aicrowd.com/challenges/global-wheat-challenge-2021/dataset_files).

### Directory Structure

```
.
├── Dockerfile 
├── EDA.ipynb                    (Exploratory Data Analysis)
├── FRCNN_Resnet_inference.py    (inference)
├── FRCNN_Resnet_training.py     (training)
├── README.md
├── datasets.py 
├── detection                    (Helper functions for training)
├── models.py 
├── my_utils.py                  (Helper functions)
├── submissions
│   └── submission.csv           (sample-submission)
├── test                         (Download test-images and save them in this folder)
└── train
    ├── train                    (Download train-images and save them in this folder)
    └── train.csv
```

### Evaluation Method
The metric used is the **Average Domain Accuracy** (ADA). Two boxes are matched if their Intersection over Union (IoU) is higher than a threshold of 0.5. Final Score by this approach is **56.2**% ADA. 


### References
Use [Filtering-Outputs](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) for filtering outputs during inference<br>
Use [TTA](https://github.com/kentaroy47/ODA-Object-Detection-ttA) for Test-time-augmentation<br>
[Pytorch-vision docs](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)<br>
[Pytorch-vision github](https://github.com/pytorch/vision)
