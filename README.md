# TODO:
2. Generate map's of varying complexetiy.
		2. Maybe add star shaped objects?
5. ~~Generate shards of of 50 paths~~
	1. ~~Training data~~
	2. ~~Testing data~~
6. ~~Training on complete data.~~
1. Try copying data using scatter, rather than for loop.
	1. Time and evaluate performance.
	2. ~~The bottleneck currently seems to be reading/writing.~~ -> solvd with multiple threads
	3. ~~Split the data into multiple shards/ 50 path shards.~~
1. ~~Get model accuracy also~~
1. Models to implement.
	1. ~~Adding label smoothing.~~
	2. ~~Using convoution networks for the patch encoding instead of fuly connected networks.~~
	3. ~~Check number of prediction classes.~~
4. ~~Training Planning model.~~
	1. ~~Using the model to decode plan.~~
4. Correct the validation data
4. Complete the extraction of the training data.
5. Position encoding of decoder.
6. Look into region proposal networks.
4. Visualize the self-attention among the patches.
4. Adding position encoding to the decoder inputs
	1. Save the current model to github.
3. Add mixed precision training (after stable model)
