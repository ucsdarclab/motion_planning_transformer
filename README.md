# TODO:
1. Set up maps, RRT* paths.
	1. ~~Map size of atmost 24x24m with a pixel resolution of 0.05m.~~
	2. Generate map's of varying complexetiy.
	3. ~~Dilate obstacles with robot shape~~
	4. ~~To check for collision, convert robot position to pixel co-ordinates.~~
2. ~~Convert maps and paths to trainable set.~~
3. Set up Transfomer model - with the Vision Transformer
	1. Loading up training data.
	2. Encoding goal and starting positions.
	3. Setting up training for termination.
	2. Setting up parameters for training.
4. Training Planning model.
	1. Using the model to decode plan.