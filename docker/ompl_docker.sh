# A script to run ompl-docker

docker run -it \
    -p $1:8800 \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v $XAUTH:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/global_planner:/root/global_planner \
    -v ~/global_planner_data:/root/global_planner_data \
    ompl-global \
    bash
