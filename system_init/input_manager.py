import argparse
import torch
import os
import datetime

import random
import string

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        filename = values[0].name
        if os.path.exists(filename):
            f = open(filename, "r")
            parser.parse_args(f.read().split(), namespace)
            f.close()
        else:
            raise(FileExistsError("File doesn't exist: {}".format(filename))) #should never get here


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


available_classifiers = ['3FC', 'simple_cnn', 'mlp', 'linear', 'wrnscat_12_32', 'wrnscat_12_16', 'wrnscat_12_8', 'wrnscat_50_2', 'wrn_16_32',
                        'wrn_16_16', 'wrn_16_8', 'wrn_16_2', 'wrn_50_2', 'wrn_50_2_torch', 'wrn_50_2_torch_pretrained','wrnscat_short_50_2']

available_scatternets = ['mallat', 'mallat_l', 'dtcwt', 'dtcwt_l']

avalable_datasets = ['mnist', 'flowers', 'tinyimagenet']


def parse_input(arg_list):
    parser = argparse.ArgumentParser(description='Scattering Networks Tests Framework')

    parser.add_argument('-env_file',  type=open, nargs=1, required=False,
                        help='File that contains the enviroment setup params', action=LoadFromFile)
    parser.add_argument('-scat_file', type=open, nargs=1, required=False,
                        help='File that contains the scattering setup params', action=LoadFromFile)
    parser.add_argument('-net_file',  type=open, nargs=1, required=False,
                        help='File that contains the dnn network setup params', action=LoadFromFile)
    parser.add_argument('-data_file', type=open, nargs=1, required=False,
                        help='File that contains the dataset setup params', action=LoadFromFile)
    parser.add_argument('-output_file', type=open, nargs=1, required=False,
                        help='File that contains the enviroment setup params', action=LoadFromFile)

    #SCAT SETUP
    parser.add_argument('-enable_scat', nargs='?', default=1, choices=(0, 1), type=int,
                        help="1 to enable Scattering, 0 to disable it")
    parser.add_argument('-scat_type', nargs='?', default='mallat', choices=available_scatternets, type=str,
                        help="To make scat nets available enable_scat argument should be set to 1")
    parser.add_argument('-scat_order', nargs='?', default=2, type=int, choices=(1, 2),
                        help="Sets the order of Mallat's scattering network")
    parser.add_argument('-J', nargs='?', default=2, choices=(0, 1, 2, 3, 4), type=int,
                        help="Allowed between 0 and 4. Be careful not all scattering nets support all choices.")
    parser.add_argument('-half_scat_feat_resolution', nargs='?', default=0, choices=(0, 1), type=int,
                        help="1 to half the resolution after scattering (average pooling), 0 to disable it. "
                             "Note: only available with DTCWT networks")

    #DNN SETUP
    parser.add_argument('-classifier', nargs='?', default='simple_cnn', type=str, choices=available_classifiers,
                        help="DNN models to chose from, can be combined with scattering nets")
    parser.add_argument('-use_scheduler', nargs='?', default=0, type=int, choices=[0, 1],
                        help="1 to enable scheduler, 0 - disable")
    parser.add_argument('-scheduler_type', nargs='?', type=str, default='onPlateau', choices=['onPlateau', 'stepLR'],
                        help="LR scheduler type.")
    parser.add_argument('-scheduler_step', nargs='?', type=int, default=-1,
                        help="Step size if stepLR is chosen. Default -1")
    parser.add_argument('-optim', nargs='?', default='sgd', type=str, choices=['sgd', 'adam'],
                        help="Choise an optimiser to use. Default: SGD")
    parser.add_argument('-wrn_dropout', nargs='?', default=0, type=int, choices=[0, 1],
                        help="Enable dropout in the ResBlocks Default: false") #TODO rename to dnn_droput
    parser.add_argument('-lr', nargs='?', default=0.01, type=float,
                        help="Defines the learning rate using during training")
    parser.add_argument('-epochs', nargs='?', default=5, type=int,
                        help="Define the number of epochs to train the model for")
    parser.add_argument('-batch_sz', nargs='?', default=128, type=int, help="Defined the batch size for the input data")

    #DATA SETUP
    parser.add_argument('-dataset', nargs='?', default="mnist", choices=avalable_datasets, type=str,
                        help="Chose a dataset")
    parser.add_argument('-augment_data', nargs='?', default=0, type=int, choices=[0, 1],
                        help="1 to enable data augmentation, 0 - disable")
    parser.add_argument('-num_workers', nargs="?", type=int, default=0,
                        help="Enable multithreaded data loading by specifying a non-zero value. Disabled by default.")
    parser.add_argument('-set_datasize', nargs='?', default=-1, type=int,
                        help="Explicitly set the size of the data for the dataset (Multi-resolution experiment)")
    parser.add_argument('-partion_tr_data', nargs='?', default=1, type=float,
                        help="A portion of training data used for training. Should be between 0 and 1.")

    #OUTPUT SETUP
    parser.add_argument('-json', nargs='?', type=str, default=None, help="Name of output Json file. If not specified, name will be generated automatically")
    parser.add_argument('-out_dir', nargs='?', type=str, default=None, help="Directory where your output files will be saved")
    parser.add_argument('-out_folder', nargs='?', type=str, default=None, help="Additional subdir folder that will be added to the out_dir")
    parser.add_argument('-checkpoints', nargs='?', type=str, default=None, help="Directory where your checkpoint files will be saved. Tries to create 2 sub directories: data and model")


    #RUNENV SETUP
    parser.add_argument('-fix_seeds', nargs='?', default=0, type=int, choices=[0, 1],
                        help="Allows to fix randomisers for reproducibility")
    parser.add_argument('-runon', nargs='?', default=None, type=str, choices=['monet', 'condor'],
                        help="Choise an environment where to run the script")
    parser.add_argument('-data_root', nargs="?", type=str, default=None,
                        help="If not running on monet or condor, speify the root dir for datasets")
    parser.add_argument('-explicit_cpu', nargs="?", type=int, default=0, choices=[0, 1],
                        help="Set 1 if want to run on cpu")
    parser.add_argument('-model_checkpoint_freq', nargs="?", type=int, default = -1,
                        help="Frequency of model checkpointing (in epochs). Disabled by default")
    parser.add_argument('-data_checkpoint_freq', nargs="?", type=int, default = -1,
                        help="Frequency of json-data checkpointing (in epochs). Disabled by default")
    parser.add_argument('-save_final_model', nargs="?", type=int, choices=[0,1], default=1,
                        help="Select 1 to save the trained model at the end of training. Disabled by default.")

    #OTHER
    parser.add_argument('-n_classes', nargs='?', default=10, type=int,
                        help="Number of classes in the input dataset, can be ignored for defined datasets")
    parser.add_argument('-in_shape', nargs='?', default=28, type=int,
                        help="Image size in the input dataset, can be ignored for defined datasets")

    print("Finished parsing input arguments.")
    return parser.parse_args(arg_list)

def fill_arguments_container(in_args, arguments_container):

    arguments_container['setup_args'].J = in_args.J
    arguments_container['setup_args'].n_classes = in_args.n_classes
    arguments_container['setup_args'].in_shape = in_args.in_shape
    arguments_container['setup_args'].batch_sz = in_args.batch_sz
    arguments_container['setup_args'].classifier = in_args.classifier
    arguments_container['setup_args'].dataset = in_args.dataset
    arguments_container['setup_args'].lr = in_args.lr
    arguments_container['setup_args'].epochs = in_args.epochs
    arguments_container['setup_args'].scat_order = in_args.scat_order
    arguments_container['setup_args'].optim = in_args.optim
    arguments_container['setup_args'].augment_data = in_args.augment_data
    arguments_container['setup_args'].wrn_dropout = True if in_args.wrn_dropout == 1 else False
    arguments_container['setup_args'].runon = in_args.runon
    arguments_container['setup_args'].set_datasize = in_args.set_datasize
    arguments_container['setup_args'].data_root = in_args.data_root
    arguments_container['setup_args'].channels_after_scaternet = 0
    arguments_container['setup_args'].in_channels = 0
    arguments_container['setup_args'].num_workers = max(0, in_args.num_workers)
    arguments_container['setup_args'].scheduler_step = in_args.scheduler_step
    arguments_container['setup_args'].half_scat_feat_resolution = True if in_args.half_scat_feat_resolution == 1 else False

    arguments_container['shared_args'].enable_scat = in_args.enable_scat
    arguments_container['shared_args'].scat_type = in_args.scat_type if in_args.enable_scat == 1 else None
    arguments_container['shared_args'].device = ''
    arguments_container['shared_args'].explicit_cpu = in_args.explicit_cpu
    arguments_container['shared_args'].classes = []
    arguments_container['shared_args'].use_scheduler = in_args.use_scheduler
    arguments_container['shared_args'].scheduler_type = in_args.scheduler_type
    arguments_container['shared_args'].json_filename = in_args.json
    arguments_container['shared_args'].checkpoint_dir = in_args.checkpoints

    arguments_container['processing_args'].fix_seeds = in_args.fix_seeds
    arguments_container['processing_args'].num_workers = in_args.num_workers
    arguments_container['processing_args'].partion_tr_data = in_args.partion_tr_data

    arguments_container['out_args'].model_checkpoint_freq = in_args.model_checkpoint_freq
    arguments_container['out_args'].data_checkpoint_freq = in_args.data_checkpoint_freq
    arguments_container['out_args'].save_final_model = True if in_args.save_final_model == 1 else False

    if arguments_container['out_args'].data_checkpoint_freq is not None or arguments_container['out_args'].model_checkpoint_freq is not None:
        arguments_container['out_args'].do_checkpoint = True

    if in_args.explicit_cpu == 1:
        arguments_container['shared_args'].device = torch.device("cpu")
    else:
        arguments_container['shared_args'].device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    std_filename = generate_std_filename(arguments_container['setup_args'].dataset,
                                                                            arguments_container['setup_args'].classifier,
                                                                            arguments_container['shared_args'].scat_type,
                                                                            arguments_container['setup_args'].batch_sz,
                                                                            arguments_container['setup_args'].epochs,
                                                                            arguments_container['processing_args'].partion_tr_data)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    std_out_folder = "OUTPUT4-" + std_filename

    final_out_folder = std_out_folder if in_args.out_folder is None else os.path.join(in_args.out_folder, std_out_folder)

    arguments_container['shared_args'].std_filename = std_filename
    arguments_container['shared_args'].out_dir = os.path.join(current_dir, final_out_folder) if in_args.out_dir is None else os.path.join(in_args.out_dir, final_out_folder)

    print("Finished filling the argument container.")

def generate_std_filename(dataset, dnn, scattype, batch_size, epochs, data_partition):
    date = datetime.datetime.now().date().__str__().replace('-','')
    time = datetime.datetime.now().time().__str__().split('.')[0].replace(":",".")
    space = "_"
    scattype = "no_scat" if scattype == None else scattype
    return date + space + time + space + dataset + space + str(int(data_partition*100.)) + "%" + space \
               + dnn + space + scattype + space + "bs_" + str(batch_size) + space \
                + "ep_" + str(epochs) + space + "uid" + space + get_random_string(6)