#!/bin/bash
# :arg[1]: name of the container
# :arg[2]: number of containers
# Stop the orchestra containers

for CID in $(seq $2 $3)
do
	docker stop data_$1_$CID
done