## SCATTERING-BASED HYBRID NETWORKS: AN EVALUATION AND DESIGN GUIDE
### This repository contains the source code for the experiments executed in the paper mentioned above and published in ICIP 2021.

### Instructions

To run experiments first download the code, create and run a virtual environment, install the required packages (specified in requirements.txt)

#### Experiments

The resources available in this framework are specified in system_init/input_manager.py:

```
available_classifiers = ['3FC', 'simple_cnn', 'mlp', 'linear', 'wrnscat_12_32', 'wrnscat_12_16', 'wrnscat_12_8', 'wrnscat_50_2', 'wrn_16_32',
                        'wrn_16_16', 'wrn_16_8', 'wrn_16_2', 'wrn_50_2', 'wrn_50_2_torch', 'wrn_50_2_torch_pretrained','wrnscat_short_50_2']

available_scatternets = ['mallat', 'mallat_l', 'dtcwt', 'dtcwt_l']

avalable_datasets = ['mnist', 'flowers', 'tinyimagenet']
```

The following classifiers are compatible with scattering networks: 


```
'wrnscat_12_32', 'wrnscat_12_16', 'wrnscat_12_8', 'wrnscat_50_2','wrnscat_short_50_2'
```

To run an experiment, please, use main.py with the following arguments:
* -env_file ----> File that contains the enviroment setup params
* -scat_file ---> File that contains the scattering setup params
* -net_file ----> File that contains the dnn network setup params
* -data_file ---> File that contains the dataset setup params
* -output_file -> File that contains the enviroment setup params

Examples of such files could be found in experiment_setup/datasets/{dataset name} for experiments performed in the presented paper.

It is possible to combine all arguments in a single file, the full list of arguments may be found in system_init/input_manager.py::parse_input() or by typing 
```
main.py -h
```

### Repository Structure

Here the basic structure and logic of the repository will be outlined. Hopefully, this will help you navigate the project.

#### Scattering Networks

Some already existing packages with scattering networks were employed in our experiments, some of which were amended.
To avoid version mismatch and other unwanted difficulties, those packages were included in the source code:

* kymatio
* pytorch_wavelets
* scatnet_learn
 
#### System Control
 
Another important part of the framework is the system control which is specified in the following two folders:

* system_init -----> Includes handling of datasets and networks
* system_actions -> Specifies train and test loops

#### Experiments Support

Other directories include: 

* experiment_setup --> Input files used to run the exeriments
* results_decoder ----> Code for interpreting the results, used to transform .json output into reports and visuals

#### Run File

* main.py -> The application file that brings the framework together and allows the running of experiments

### Citation

Please use the following form when citing this work:

```
@inproceedings{minskiy2021scattering,
  title={Scattering-Based Hybrid Networks: An Evaluation and Design Guide},
  author={Minskiy, Dmitry and Bober, Miroslaw},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2793--2797},
  year={2021},
  organization={IEEE}
}
```

