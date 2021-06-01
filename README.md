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
To generate training or validation data set you can run the following command:

```
python3 rrt_star_map.py --start=... --numEnv=... --envType=... --numPaths=... --fileDir=... --mapFile
```

To understand what each of these arguments stand for, run `python3 rrt_star_map.py --help`.

You can download the training and validation data for maze and forest environments from [here](https://drive.google.com/file/d/1ciyn4_kDGEgVFWAAaKrfPqVme1U6vw9N/view?usp=sharing) and [here](https://drive.google.com/file/d/14-2oEtf4u9bLt6JwGezb0iqCS9_VzS8Y/view?usp=sharing) respectively.

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

### Contributing

