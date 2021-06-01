FROM ompl_bionic as BUILDER

FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04 AS BASE

COPY --from=BUILDER /usr/local/include/ompl /usr/local/include/ompl
COPY --from=BUILDER /usr/local/lib/libompl* /usr/local/lib/
COPY --from=BUILDER /usr/local/share/ompl /usr/local/share/ompl
COPY --from=BUILDER /usr/local/bin/ompl_benchmark_statistics.py /usr/local/bin/ompl_benchmark_statistics.py
COPY --from=BUILDER /usr/local/share/man/man1/ompl_benchmark_statistics.1 /usr/local/share/man/man1/ompl_benchmark_statistics.1
COPY --from=BUILDER /usr/local/share/man/man1/plannerarena.1 /usr/local/share/man/man1/plannerarena.1
COPY --from=BUILDER /root/ompl /root/ompl
COPY --from=BUILDER /usr/local/lib/pkgconfig/ompl.pc /usr/local/lib/pkgconfig/ompl.pc
COPY --from=BUILDER /usr/lib/python3/dist-packages/ompl /usr/lib/python3/dist-packages/ompl

# Files required for OMPL
RUN apt-get update && apt-get install -y \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-numpy-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    libboost-test-dev \
    libflann-dev \
    libode-dev \
    libeigen3-dev \
	python3-pip\
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install jupyter \
				einops \
				scikit-image \
                webdataset \
                tensorboard \
                tqdm\
                toolz

#RUN apt-get update && apt-get install -y \

WORKDIR /workspace