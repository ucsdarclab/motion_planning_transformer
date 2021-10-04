#!/bin/bash
# A script to run ompl-docker
SAMPLES=50
for CID in {0..19}
do
	docker run -d\
	    --rm \
	    --shm-size="2g"\
	    -v ~/global_planner:/workspace \
	    -v ~/global_planner_data:/root/data \
	    mpt \
	    python3 rrt_star_map.py \
			--start=$((CID*SAMPLES)) \
			--numEnv=$SAMPLES \
			--envType=maze \
			--fileDir=/root/data/maze4/val780 \
			--numPaths=1\
			--height=710 \
			--width=710		
done