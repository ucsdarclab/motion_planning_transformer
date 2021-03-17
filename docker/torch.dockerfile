FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN pip install jupyter \
				einops \
				scikit-image

RUN pip install webdataset
RUN pip install tensorboard