FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN pip install   pytorch-lightning
RUN pip install -U git+https://github.com/albu/albumentations --no-cache-dir
RUN pip install --upgrade albumentations 
RUN pip install timm
RUN pip install odach
RUN pip install ensemble_boxes
RUN pip install opencv-python-headless
RUN pip install --no-cache-dir --upgrade pip

RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN apt install -y p7zip-full p7zip-rar

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
