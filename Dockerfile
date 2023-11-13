FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 rsync -y
RUN conda install grpcio
RUN python --version
RUN pip install protobuf
RUN pip install \
    pytorch-lightning==1.1.3 \
    lightning-bolts==0.3.0 \
    seaborn \
	wandb \
    pydevd-pycharm~=212.5712.39 \
    opencv-python \
    gym \
    tqdm==4.36.1 \
    six==1.12.0 \
    tensorboard==2.4.0 \
    numpy==1.17.3 \
    matplotlib==3.5.2 \
    pandas==1.0.3 \
    Pillow==9.1.1 \
    scikit_learn==1.0.2 \
    scipy==1.7.3 \
    timm==0.5.4 \
    torch==1.10.0 \
    torchvision==0.11.1

RUN apt-get install vim

