# Imitrob Dataset Code



## General information

This is a supplementary code for the Imitrob dataset located at <http://imitrob.ciirc.cvut.cz/imitrobdataset.php>. The code and the dataset are intended to be used for 6D pose estimator training and evaluation (benchmarking).


## Installation  

Simply clone this repo, install the required Python packages (listed below) and download the necessary data.  

### Requirements  

* Python packages required to run the code are listed in the [requirements.txt](requirements.txt) file. All the required packages can be installed by:  
`$ pip install -r requirements.txt`  
* Training a 6D pose estimator requires the [Imitrob Train](http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) dataset ("light" version is also available) and a dataset of background images (for augmentation), e.g., the ImageNet or a subset of it (you can use [mini-ImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools) to generate the subset).  
* Evaluation requires the [Imitrob Test](http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) dataset.  
* The scripts were tested on Ubuntu 18.04 with Python 3.6. Nonetheless, they should run on most platforms where PyTorch can be installed and with Python version 3.6 or later.


## Usage

The base class for working with the Imitrob dataset is `imitrob_dataset` in [imitrob_dataset.py](imitrob_dataset.py) and it is based on the [PyTorch](https://pytorch.org/docs/stable/data.html) `Dataset` class. As such, it can be easily used to train any PyTorch model via the `DataLoader` class.  

The Imitrob dataset specific parameters are described in the `imitrob_dataset` constructor. For general PyTorch dataset parameters  and usage, please see the [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class documentation.  

The [trainer.py](trainer.py) file contains an example of using the dataset with the [DOPE](https://github.com/NVlabs/Deep_Object_Pose) pose estimator. The [evaluation.py](evaluation.py) file is used to evaluate an already trained network. Use the training and evaluation scripts as a template to train and test your estimator. In case of a PyTorch model, you will typically only need to assign an instance of your model to the `net` variable, i.e., changing [this line](trainer.py#L363) of the trainer.py file:
```python
net = dope_net(lr, gpu_device)  # switch dope_net for your own network
```

Similar change has to be done in the evaluation.py script.

The data acquisition process is described in [this readme file](data_acquisition/README.md).

### Training

The [trainer.py](trainer.py) file performs training (and the evaluation after training, though this can be skipped). The script accepts command line arguments to specify which parts of the dataset should be used. Use help invocation command to see all the possible options and their descriptions:  
```bash
$ python trainer.py -h
```

Here is an example of executing the trainer.py script with a condensed summary of the options:  
```bash
$ python trainer.py --traindata "path/to/train/data/directory"
                    --testdata "path/to/test/data/directory"
                    --bg_path "path/to/baground/data/directory"
                    --exp_name experiment_1
                    --randomizer_mode overlay
                    --gpu_device 0
                    --dataset_type roller
                    --subject [S1,S2,S3]
                    --camera [C1,C2]
                    --hand [LH,RH]
                    --subject_test [S4]
                    --camera_test [C1,C2]
                    --hand_test [LH,RH]
                    --task_test [clutter,round,sweep,press,frame,sparsewave,densewave]
```

The list notation (multiple items in square brackets, e.g., `[XX,YY,ZZ]`) is used to specify multiple options for a given argument. For example, `--camera [C1,C2]` tells the script to use images from both the `C1` and `C2` cameras.

### Evaluation

The [evaluation.py](evaluation.py) file performs the estimator evaluation. Typically, the trainer.py script is used to both train and evaluate the estimator. However, the evaluation can also be performed separately. Similar to the trainer.py, dataset and network settings are passed to the script via the command line. The first positional argument is the path to the train weights of the network (typically `directory/something.pth.tar`). Run the script with the `-h` argument to see the list and description of possible arguments:  
```bash
$ python evaluation.py -h
```

We provide example weights for DOPE [here](https://data.ciirc.cvut.cz/public/groups/incognite/Imitrob/test_net_weights.zip). The weights are trained for the _glue gun_ tool only, using the following configuration:  
```bash
$ python trainer.py --traindata "Imitrob\Train" --testdata "Imitrob\Test" --bg_path "mini_imagenet_dataset\images" --epochs 5 --exp_name experiment_5 --randomizer_mode overlay --gpu_device 0 --dataset_type gluegun --subject [S1,S2,S3,S4] --camera [C1,C2] --hand [LH,RH] --subject_test [S1,S2,S3,S4] --camera_test [C1,C2] --hand_test [LH,RH] --task_test [clutter,round,sweep,press,frame,sparsewave,densewave]
```  
These can be used to run the evaluation.py script.

If you develop and evaluate your own model on our dataset, we would really appreciate if you send us your results. We will include them in the leaderboard below.

| Method | training configuration | testing configuration | metric | results |
|:---:|:---|:---|:---:|---:|
| DOPE (original)  | full ImitrobTrain | full ImitrobTest | ADD5 |   |
|   |   |   |   |   |
|   |   |   |   |   |


## Dataset extension - new tools

[Data acquisition tutorial](data_acquisition/README.md) (codes, docker, sample data)

We provide methods for the acquisition of training data for a new tool.

First, mount a tracker on the tool and prepare a tracing tool with mounted tracker.

Afterwards follow the steps to calibrate the data acquisition setup as well as calibrating the bounding box for the tool.

<img src="data_acquisition/trace-extractor/images/trace_workflow.png" width="1000"/>
-

Now you can  record data for the manipulated object while manipulating it in front of the green background. 6DoF positions are extracted and objects segmentation masks are created.  

https://user-images.githubusercontent.com/17249817/185711525-d843e1ba-f15c-4c0c-bc9c-3b83eaa505a7.mp4

Individual steps with corresponding codes and sample data are described in [Data acquisition tutorial](data_acquisition/README.md).

## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

If you use this code in your research, please, give us an attribution, using the following citation:


```BibTex
 @Misc{imitrobdataset,
  author =   {{CIIRC CTU in Prague}},
  title =    {{I}mitrob dataset version 2.0},
  howpublished = {\url{http://imitrob.ciirc.cvut.cz/imitrobdataset.php}},
  year =     2022
}
```


## Acknowledgment

Part of this work is based on the code of [NVidia Deep Object Pose](https://github.com/NVlabs/Deep_Object_Pose) ([paper](https://arxiv.org/abs/1809.10790)).
We also used the [mini-ImageNet library](https://github.com/yaoyao-liu/mini-imagenet-tools) (proposed in the paper [Matching Networks for One Shot Learning](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf)) to generate the background images in our benchmarks.  

## Contact

Manager of the dataset: &#107;&#097;&#114;&#108;&#097;&#046;&#115;&#116;&#101;&#112;&#097;&#110;&#111;&#118;&#097;&#064;&#099;&#118;&#117;&#116;&#046;&#099;&#122;.
