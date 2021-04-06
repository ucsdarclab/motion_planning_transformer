#!/bin/bash
# A script to run ompl-docker
SAMPLES=400
for CID in {0..24..1}
do
	docker run -d \
	    --rm \
	    --name=data_$1_$CID \
	    --shm-size="2g"\
	    -e DISPLAY=$DISPLAY \
	    -e QT_X11_NO_MITSHM=1 \
	    -v $XAUTH:/root/.Xauthority \
	    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	    -v ~/global_planner:/root/global_planner \
	    -v ~/global_planner_data:/root/global_planner_data \
	    ompl-global \
	    python3 rrt_star_map.py $((CID*SAMPLES)) $SAMPLES
done