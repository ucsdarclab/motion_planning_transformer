# TODO:
2. Generate map's of varying complexetiy.
		2. Maybe add star shaped objects?
3. Set up architecture for MHA for just encoder model.
	1. Model architecture.
		1. ~~Function to map attention layer outputs to input map regions.~~
	2. ~~Set up training data.~~
	3. ~~Set up cost function.~~
		1. ~~Modify code to make predictions~~
	2. ~~Model start and goal position.~~
	3. ~~Position Encoding for larger images.~~
4. ~~Extracting path from the given set of predictions.~~
5. ~~Calculating the probability of existence of path from the given map.~~
6. Planning on a map with different sizes.
	1. ~~Write a new position encoder~~
	2. ~~Test the position encoder.~~
7. ~~Train the dataset.~~
8. Testing on totally different maps with similar obstacles:
	1. Evaluate model5 on totally different map with same size.
	2. Maybe test model4 also?
4. Visualize the self-attention among the patches.
3. Add mixed precision training (after stable model)