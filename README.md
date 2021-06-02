# Motion Planning Transformers: One Model to Plan them All
The network architecture for Motion Planning Transformers (MPT).

![Transformer Figure](transformer_fig.jpg)

### Requirements
All our experiments were conducted on `Ubuntu18.04` with `Python3.6`. To generate the data, and evaluate the planner you will need the [OMPL-1.4.2](https://ompl.kavrakilab.org/index.html) library with the Python bindings.

Other python dependencies are given in `requirements.txt`. You can install the package using pip:

```
pip3 install -r requirments.txt
```

#### Using Docker

We highly recommend that to replicate our testing environment, users can use our docker container which contains all the necessary libraries packages. Download the `.tar` [file](https://drive.google.com/file/d/154E338PduQPHfO0sUqA8ZST1GaQodY41/view?usp=sharing).

To load the image from the `tar` file, run the following:

```
docker load -i mpt_container.tar
```

To run the container, run the following command:

```
docker run -it --gpus all --shm-size="16G" -v ~/global_planner_data:/root/data
```

### Creating Dataset
To generate training or validation data set for the point environment you can run the following command:

```
python3 rrt_star_map.py --start=... --numEnv=... --envType=... --numPaths=... --fileDir=... --mapFile
```

To collect data samples for the car environment you can run the following command:

```
python3 sst_map.py --start=... --numEnv=... --numPaths=... --fileDir=...
```

You can download all the data we used for training from [here]().

### Training

To train the data, run the following command:

```
python3 train.py --batchSize=... --mazeDir=... --forestDir=... --fileDir=...
```

### Evaluation

To evaluate a set of validation paths, you can run the following code:

```
python3 eval_model.py --modelFolder=... --valDataFolder=... --start=... --numEnv=... --epoch=... --numPaths=...
```

### Pre-trained Models
You can download the pretrained models for the point robot and Dubins Car Model from [here]() and [here]().

### Results

<table>
	<thead>
	<tr>
		<th>Environment </th>
		<th colspan="3">Random Forest</th>
		<th colspan="3">Maze</th>
	</tr>
	</thead>
	<tbody>
		<tr>
			<td></td>
			<td>Accuracy</td>
			<td>Time (sec)</td>
			<td>Vertices</td>
			<td>Accuracy</td>
			<td>Time (sec)</td>
			<td>Vertices</td>
		</tr>
		<tr>
			<td>RRT*</td>
			<td>99.88\%</td>
			<td>5.44</td>
			<td>3227.5</td>
			<td>100\%</td>
			<td>5.36</td>
			<td>2042</td>
		</tr>
		<tr>
			<td>IRRT*</td>
			<td>99.88\%</td>
			<td>0.42</td>
			<td>267</td>
			<td>100\%</td>
			<td>3.13</td>
			<td>1393.5</td>
		</tr>
		<tr>
			<td>MPT-RRT*</td>
			<td>97.68\%</td>
			<td>0.20</td>
			<td>251</td>
			<td>98.96\%</td>
			<td>0.83</td>
			<td>615</td>
		</tr>
		<tr>
			<td>MPT-IRRT*</td>
			<td>97.68\%</td>
			<td>0.07</td>
			<td>133</td>
			<td>98.96\%</td>
			<td>0.74</td>
			<td>557</td>
		</tr>
	</tbody>
</table>

### Contributing

This code base is currently for reviewing purpose alone. Please do not distribute.