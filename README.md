# Imitrob Dataset Code



## General information

This is a supplementary code for the Imitrob dataset located at <http://imitrob.ciirc.cvut.cz/imitrobdataset.php>.


## Installation  

Simply cone this repo and install the required Python packages and download the necessary data.  

### Requirements  

* Python packages required to run the code are listed in the [requirements.txt](requirements.txt) file. All required packages can be installed by:  
`pip install -r requirements.txt`  
* Training a 6D pose estimator requires the [Imitrob Train](http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) dataset and a dataset of background images (for augmentation), e.g., the ImageNet or a subset of it - [mini-ImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools).  
* Evaluation requires the [Imitrob Test](http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) dataset.  
* The scripts were tested on Ubuntu 18.04 and with Python 3.6. Nonetheless, they should run on any platform where PyTorch can be installed and with Python version 3.6 or later.


## Usage

The base class for working with the Imitrob dataset is `imitrob_dataset` in [imitrob_dataset.py](imitrob_dataset.py) and it is based on the [PyTorch](https://pytorch.org/docs/stable/data.html) `Dataset` class. As such, it can be easily used to train any PyTorch model via the `DataLoader` class.  

The [trainer.py](trainer.py) file contains an example of using the dataset with the [DOPE](https://github.com/NVlabs/Deep_Object_Pose) pose estimator. The [evaluation.py](evaluation.py) file is used to evaluate an already trained network. Use the training and evaluation scripts as a template to train and test your estimator. In case of a PyTorch model, you will typically only need to assign an instance of your model to the `net` variable, i.e., changing [this line](trainer.py#L361) of the trainer.py file:
```python
net = dope_net(lr, gpu_device)  # switch dope_net for your own network
```

### Training

The [trainer.py](trainer.py) file performs training (and the evaluation after training, though this can be skipped). The script accepts command line arguments to specify which parts of the dataset should be used. Use this command to see all the possible options and their descriptions:  
```bash
$ python trainer.py -h
```

Here is a condensed summary of the options:  
```bash
$ python trainer.py --traindata "path/to/train/data/directory" --testdata "path/to/test/data/directory" --bg_path "path/to/baground/data/directory" --exp_name experiment_1 --randomizer_mode overlay --gpu_device 0 --dataset_type roller --subject [S1,S2,S3] --camera [C1,C2] --hand [LH,RH] --subject_test [S4] --camera_test [C1,C2] --hand_test [LH,RH] --task_test [clutter,round,sweep,press,frame,sparsewave,densewave]
```

### Evaluation

The [evaluation.py](evaluation.py) file performs the estimator evaluation. Typically, the trainer.py is used to both train and evaluate the estimator. However, the evaluation can also be performed separately. Similar to the trainer.py, dataset and network settings are passed to the script via a command line. Run the script with a `-h` argument to see the list and description of possible arguments:  
```bash
$ python evaluation.py -h
```

## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

If you use this code in your research, please, give as an attribution, using the following citation:

  
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

## Contact

Manager of the dataset: &#107;&#097;&#114;&#108;&#097;&#046;&#115;&#116;&#101;&#112;&#097;&#110;&#111;&#118;&#097;&#064;&#099;&#118;&#117;&#116;&#046;&#099;&#122;.