#!/bin/bash
# A script to run ompl-docker
SAMPLES=50

for CID in $(seq $2 $3)
do
	docker run -d \
	    --rm \
	    --name=data_$1_$CID \
	    --shm-size="2g"\
	    -v ~/global_planner:/workspace \
	    -v ~/global_planner_data:/root/data \
	    pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter\
	    python generateShards.py $CID $((CID*SAMPLES)) $SAMPLES
done