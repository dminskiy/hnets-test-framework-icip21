import torch.nn as nn
import torch
from .models.CIFAR.models_simple import *
from .models.CIFAR.models_scatterNet import *
from .models.CIFAR.models_wrn import *
from .models.IMAGENET.models_scatterNet import wrnscat_50_2, wrnscat_short_50_2
from .models.IMAGENET.models_wrn import *
import math
import numpy as np

from kymatio.torch import Scattering2D
from scatnet_learn import ScatLayerj1, InvariantLayerj1
from system_init.my_layers.mallat import helper_funcs
from system_init.my_layers.mallat.mallat_mixing import Scattering2DMixed
from collections import OrderedDict
import torchvision.models as models


'''
Sets and returns: 
    model     - can be adjust using input parameters
    optimiser - SGD/Adam are supported, lr is adjustable with input parameters (scheduler may be used to change it dynamically)
    scheduler - all default parameters are used there
    criterion - cross entropy loss is used in all the cases 
    
    setup_argumets [in] - carries all the information required to setup the model 
    processing_argumets [out] - carries all the model information 
'''
def build_model(processing_arguments, setup_arguments):
    if processing_arguments.shared_arguments.enable_scat == 1:
        get_scatterNet(processing_arguments, setup_arguments)
        get_model(processing_arguments, setup_arguments)
    else:
        get_model(processing_arguments, setup_arguments)
        processing_arguments.scatNet = None

    get_optimiser(processing_arguments, setup_arguments)

    if processing_arguments.shared_arguments.use_scheduler == 1:
        get_scheduler(processing_arguments, setup_arguments)
    else:
        processing_arguments.scheduler = None

    get_criterion(processing_arguments)

    if processing_arguments.fix_seeds == 1:
        fix_seeds(0)

def fix_seeds(num):
    torch.manual_seed(num)
    torch.backends.cudnn.deterministic = True #TODO fix it
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

#using the same optimiser for everything at the moment, can be changed later
def get_optimiser(processing_arguments,setup_arguments):
    optim = setup_arguments.optim
    lr = setup_arguments.lr
    model = processing_arguments.model

    if optim == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optim == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    else:
        raise ValueError("Optimiser is not supported: {}".format(optim))

    processing_arguments.optimiser = optimiser

def get_scheduler(processing_arguments, setup_argumetns):
    scheduler_type = setup_argumetns.shared_arguments.scheduler_type
    scheduler_step = setup_argumetns.scheduler_step
    optimiser = processing_arguments.optimiser

    if scheduler_type == 'onPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
    elif scheduler_type == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=scheduler_step, gamma=0.1)
    else:
        raise(ValueError("This scheduler is not supported yet: ". format(scheduler_type)))

    processing_arguments.scheduler = scheduler

def get_criterion(processing_arguments):
    criterion = nn.CrossEntropyLoss()
    processing_arguments.criterion = criterion

def get_classifier(setup_arguments, shape_after_scatter, channels_after_scatter):

    classifier_name = setup_arguments.classifier
    n_classes = setup_arguments.n_classes
    in_shape = setup_arguments.in_shape
    in_channels = setup_arguments.in_channels
    wrn_dropout = setup_arguments.wrn_dropout

    if classifier_name == "simple_cnn":
        classifier = OneConvLayerNet(shape_after_scatter, channels_after_scatter, n_classes, conv_channels=64)
    elif classifier_name == "wrnscat_12_32":
        classifier = resnet12_32_scat(shape_after_scatter, channels_after_scatter, n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrnscat_12_16":
        classifier = resnet12_16_scat(shape_after_scatter, channels_after_scatter, n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrnscat_12_8":
        classifier = resnet12_8_scat(shape_after_scatter, channels_after_scatter, n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrn_16_32":
        classifier = resnet16_32(in_channels, n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrn_16_16":
        classifier = resnet16_16(in_channels,n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrn_16_8":
        classifier = resnet16_8(in_channels,n_classes, dropout=wrn_dropout)
    elif classifier_name == "wrn_16_2":
        classifier = resnet16_2(in_channels,n_classes, dropout=wrn_dropout)
    elif classifier_name == "3FC":
        classifier = ThreeFullyConnectedLayers(shape_after_scatter, channels_after_scatter, n_classes)
    elif classifier_name == "wrn_50_2":
        classifier = resnet50_2(in_shape,in_channels,n_classes)
    elif classifier_name == "wrn_50_2_torch":
        classifier = models.wide_resnet50_2() #TODO fix it
    elif classifier_name == "wrn_50_2_torch_pretrained":
        classifier = models.wide_resnet50_2(pretrained=True) #TODO fix it
    elif classifier_name == "wrnscat_50_2":
        classifier = wrnscat_50_2(shape_after_scatter, channels_after_scatter, n_classes)
    elif classifier_name == "wrnscat_short_50_2":
        classifier = wrnscat_short_50_2(shape_after_scatter, channels_after_scatter, n_classes)
    else:
        raise ValueError("Classifier is not yet supported. Available options: simple_cnn, wrnscat_12_16, wrnscat_12_8,"
                         " wrn_16_16, wrn_16_8, wrn_16_2 \nInput value: {}".format(classifier_name))

    return classifier

def get_model(processing_arguments, setup_arguments):

    enable_scat = setup_arguments.shared_arguments.enable_scat
    scat_type = setup_arguments.shared_arguments.scat_type
    in_shape = setup_arguments.in_shape
    in_channels = setup_arguments.in_channels

    scat_net_2_seq_merge = ['dtcwt_l', 'dtcwt', 'mallat', 'mallat_l']

    if enable_scat == 1:
        size_after_scat, channels_after_scat = shape_after_scattering(setup_arguments)

        if scat_type in scat_net_2_seq_merge:
            classifier = get_classifier(setup_arguments, size_after_scat, channels_after_scat)
            model = nn.Sequential(processing_arguments.scatNet, classifier)
        else:
            raise(ValueError("Error building a model. Scatter Net is not on the list: {}".format(scat_type)))
    else:
        model = get_classifier(setup_arguments, in_shape, in_channels)

    init_model(model)

    cuda_devices_available = torch.cuda.device_count()

    model.to(setup_arguments.shared_arguments.device)

    if setup_arguments.shared_arguments.device != torch.device("cpu"):
        if cuda_devices_available > 1:
            device_ids = list(range(cuda_devices_available))
            model = nn.DataParallel(model, device_ids=device_ids)
            setup_arguments.shared_arguments.num_devices_used = cuda_devices_available

            #get names of all devices
            all_devices_by_name = ""
            for x in device_ids:
                all_devices_by_name += torch.cuda.get_device_name(x) + " "
            setup_arguments.shared_arguments.device_name = all_devices_by_name
        else:
            #if only 1 GPU is available
            setup_arguments.shared_arguments.num_devices_used = 1
            setup_arguments.shared_arguments.device_name = torch.cuda.get_device_name(0)
    elif setup_arguments.shared_arguments.device == torch.device("cpu"):
        setup_arguments.shared_arguments.num_devices_used = 0
        setup_arguments.shared_arguments.device_name = "cpu"
    else:
        raise RuntimeError("Cannot assign running devices. Please, check your settings")

    processing_arguments.model = model

def get_scatterNet(processing_arguments, setup_arguments):

    scat_type = setup_arguments.shared_arguments.scat_type
    J = setup_arguments.J
    scat_order = setup_arguments.scat_order
    in_shape = setup_arguments.in_shape
    in_channels = setup_arguments.in_channels

    scat_post_avpool_kernel_size = None
    scat_post_avpool_stride = 1
    scat_post_avpool_padding = 0

    scatNet = None

    dtcwt_fam = ["dtcwt", "dtcwt_l"]
    malalt_fam = ["mallat", "mallat_l"]

    if setup_arguments.shared_arguments.enable_scat == 1:
        if scat_type == 'mallat':
            scatNet = Scattering2D(J=J, shape=(in_shape, in_shape),max_order=scat_order)

        elif scat_type == 'mallat_l':
            scatNet = Scattering2DMixed(in_channels=in_channels, J=J, shape=(in_shape, in_shape), max_order=scat_order,
                                        L=8, k=1, alpha=None)

        elif scat_type == 'dtcwt':
            if J == 1:
                if scat_order == 1:
                    scatNet = nn.Sequential(OrderedDict([
                        ('order1', ScatLayerj1(2))
                    ]))
                    scat_post_avpool_kernel_size = in_shape / 4 + 1 #use half of scat output +1 to half the feature resolution
                elif scat_order == 2:
                    scatNet = nn.Sequential(OrderedDict([
                        ('order1', ScatLayerj1(2)),
                        ('order2', ScatLayerj1(2))
                    ]))
                    scat_post_avpool_kernel_size = in_shape / 8 + 1 #use half of scat output +1 to half the feature resolution
                else:
                    raise ValueError("Scattering order of 1 and 2 only available for this implementation of DTWCT")
            else:
                raise ValueError("J can only be 1 in the current implementation of DTWCT")

        elif scat_type == 'dtcwt_l':
            if J == 1:
                if scat_order == 1:
                    scatNet = nn.Sequential(OrderedDict([
                        ('order1', InvariantLayerj1(in_channels))
                    ]))
                    scat_post_avpool_kernel_size = in_shape / 4 + 1 # use half of scat output +1 to half the feature resolution
                elif scat_order == 2:
                    scatNet = nn.Sequential(OrderedDict([
                        ('order1', InvariantLayerj1(in_channels)),
                        ('order2', InvariantLayerj1(7*in_channels))
                    ]))
                    scat_post_avpool_kernel_size = in_shape / 8 + 1 # use half of scat output +1 to half the feature resolution
            else:
                raise ValueError("J can only be 1 in the current implementation of DTWCT")

        if scat_type in dtcwt_fam \
            and scat_post_avpool_kernel_size is not None\
            and setup_arguments.half_scat_feat_resolution:

            scat_post_avpool_kernel_size = int(scat_post_avpool_kernel_size)
            scatNet = nn.Sequential(
                OrderedDict([
                    ('scatnet', scatNet),
                    ('post_scat_pool', nn.AvgPool2d(scat_post_avpool_kernel_size, stride=scat_post_avpool_stride,
                                                    padding=scat_post_avpool_padding))
                ])
            )

    processing_arguments.scatNet = scatNet


def shape_after_scattering(setup_arguments):

    scat_type = setup_arguments.shared_arguments.scat_type
    J = setup_arguments.J
    scat_order = setup_arguments.scat_order
    in_shape = setup_arguments.in_shape
    in_channels = setup_arguments.in_channels

    size = in_shape / (2**(J))

    if scat_type == 'dtcwt_l' or scat_type == 'dtcwt':
        channels = 7**scat_order * in_channels
        if scat_order == 2:
            size = size / (2**(J))

    elif scat_type == 'mallat' or scat_type == 'mallat_l':
        channels = helper_funcs.calculate_channels_after_scat(in_channels, J, scat_order, L=8)

    elif scat_type == 'gabor':
        #TODO Add Czaja's net filer calculations. Net not supported yet
        raise (ValueError("This type of filters is not supported yet: gabor"))

    else:
        raise ValueError("Can't get after scattering shape, scat_type isn't supported: {}".format(scat_type))

    if setup_arguments.half_scat_feat_resolution:
        size /= 2

    size = int(size + 0.5)

    return size, int(channels)

def init_model(model):
    #initialize
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, 2./math.sqrt(n))
            #m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 2./math.sqrt(m.in_features))
            m.bias.data.zero_()