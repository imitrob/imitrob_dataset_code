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

## Usage

### Training

### Evaluation


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