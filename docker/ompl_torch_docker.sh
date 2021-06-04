# A script to run mpt docker

docker run -it \
    -p 8888:8888 \
    --gpus all \
    --shm-size="16g"\
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v $XAUTH:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/global_planner:/workspace \
    -v ~/global_planner_data:/root/data \
	mpt:latest \
    bash
