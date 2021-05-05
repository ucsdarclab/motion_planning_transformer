# TODO:
1. Training model for the maze environment:
	1. ~~Train model with higher Gamma value.~~
	2. Model15 - gamma value of 4.
	3. Model16 - gamma value of 6.
2. Check training and testing code.
	1. ~~Correct the patch encoding code.~~
3. ~~Set up code on unicorn2~~
2. Training with hard mining:
	1. ~~Visualize the distribution of path patches to background~~
	3. Validate training/planning code
	4. Evaluate positional encoding
	4. Set up hard mining training
	5. Different training strategy
3. Try models with different depths
	1. ~~With 8 layers~~
	2. ~~With 4 layers~~
4. ~~Remove position encoding, and test to see if we obtain similar results.~~
3. Evaluate the model 15, 16 models for planning	
3. Investigate the reason behind correlation in the dataset.
4. Using some form of self-supervision to train network.
	1. Maps can be rotated/cropped and there should be no change in planned path.
6. Planning on a map with different sizes.
4. Visualize the self-attention among the patches.