#!/bin/bash
# A script to run ompl-docker
SAMPLES=20
for CID in {0..14..1}
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
	    -v ~/global_planner_data:/root/data \
	    ompl-global \
	    python3 rrt_star_map.py \
			--start=$((CID*SAMPLES)) \
			--samples=$SAMPLES \
			--envType=maze \
			--fileDir=/root/data/maze2/train \
			--numPaths=25 
done