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
	2. Using convoution networks for the decoder instead of fully connected networks.
4. ~~Training Planning model.~~
	1. ~~Using the model to decode plan.~~
3. Add mixed precision training, and check for improvements.
4. Visualize the self-attention among the patches.
4. Adding position encoding to the decoder inputs
	1. Save the current model to github.