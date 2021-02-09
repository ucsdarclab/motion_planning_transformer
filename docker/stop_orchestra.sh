#!/bin/bash
# :arg[1]: name of the container
# :arg[2]: number of containers
# Stop the orchestra containers

for CID in {0..$2..1}
do
	docker stop data_$1_$CID
done