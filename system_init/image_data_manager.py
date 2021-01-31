from torchvision import datasets
import torchvision.transforms as transforms
import torch
import os.path as path
import os
from .datasets.ImageNetLoader import ImageNet
from .datasets.tinyImageNet.prepare_tinyimagenet import prepare_tinyimagenet


def augment_data(setup_arguments):
    mnist_fam = ['mnist']
    current_dataset = setup_arguments.dataset
    set_datasize = setup_arguments.set_datasize

    if setup_arguments.augment_data == 1:
        if current_dataset == 'tinyimagenet':

            if set_datasize > 0:
                im_size = set_datasize
            else:
                im_size = 64

            transform_train = transforms.Compose([
                transforms.RandomCrop(im_size, padding=4),
                #transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            if set_datasize > 0:
                transform_test = transforms.Compose([
                    transforms.Resize(size=im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])

            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])

        elif current_dataset in mnist_fam:
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

        elif current_dataset == 'flowers':

            if set_datasize > 0:
                resize = int(set_datasize * 1.4)
                im_size = set_datasize
            else:
                resize = 256
                im_size =224

            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=resize, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=im_size),  # Image net standards = 224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ])
            transform_test = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(size=im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            print("WARNING! Data transformation isn't supported for this dataset: {}".format(current_dataset))
            transform_train = transforms.ToTensor()
            transform_test = transforms.ToTensor()
    else:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1)
        ])

        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1)
        ])

    return transform_train, transform_test


def load_ds_params(classes, shape, channels, setup_args):
    setup_args.n_classes = classes
    setup_args.in_shape = shape
    setup_args.in_channels = channels
    l = [i for i in range(classes)]
    setup_args.shared_arguments.classes = l


def prepare_image_data(setup_arguments, processing_arguments):
    print("Start preparing data.")

    current_dataset = setup_arguments.dataset
    set_datasize = setup_arguments.set_datasize

    imageNetRoot = "/hdd1/datasets/ImageNet/ILSVRC2012" #TODO: move to a better place

    datasets_root_dir = ''
    if setup_arguments.runon == 'condor':
        datasets_root_dir = '/user/HS221/dm00314/Desktop/'
    elif setup_arguments.runon == 'monet':
        datasets_root_dir = '/mnt/hdd1/datasets/'
    else:
        if setup_arguments.data_root is not None:
            datasets_root_dir = setup_arguments.data_root
        else:
            raise(ValueError("Dataset storage path was not specified. Please, specify."))

    if not path.exists(datasets_root_dir):
        raise(ValueError("Dataset storage path was not found: {}".format(datasets_root_dir)))

    transform_train, transform_test = augment_data(setup_arguments)

    mnist_datasets_dir = path.join(datasets_root_dir, "MNIST_datasets")

    if current_dataset == 'mnist':
        original_mnist_dir = path.join(mnist_datasets_dir, "MNIST")
        trainset = datasets.EMNIST(root= original_mnist_dir, train=True, split='mnist', transform=transform_train,
                                   download=True)
        testset = datasets.EMNIST(root= original_mnist_dir, train=False, split='mnist', transform=transform_test,
                                  download=True)

        load_ds_params(10, 28, 1, setup_arguments)

    elif current_dataset == 'flowers':
        data_dir = path.join(datasets_root_dir, 'flowers_102')

        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        trainset = datasets.ImageFolder(root=train_dir, transform = transform_train)
        testset = datasets.ImageFolder(root=test_dir, transform=transform_test)

        if set_datasize > 0:
            im_size = set_datasize
        else:
            im_size = 224

        load_ds_params(102, im_size, 3, setup_arguments)

    elif current_dataset == 'tinyimagenet':
        data_dir = path.join(datasets_root_dir, 'TinyImageNet/tiny-imagenet-200')

        if not path.exists(data_dir):
            tinyimagenet_root_dir = path.join(datasets_root_dir, 'TinyImageNet')
            os.makedirs(tinyimagenet_root_dir)
            prepare_tinyimagenet(tinyimagenet_root_dir)

        train_dir = data_dir + '/train'
        test_dir = data_dir + '/test'

        trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
        testset = datasets.ImageFolder(root=test_dir, transform=transform_test)

        if set_datasize > 0:
            im_size = set_datasize
        else:
            im_size = 64

        load_ds_params(classes=200, shape=im_size, channels=3, setup_args=setup_arguments)

    else:
        raise ValueError("Dataset is not supported. Input value: {}".format(current_dataset))

    processing_arguments.trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=setup_arguments.batch_sz,
                                                                   shuffle=True, pin_memory=torch.cuda.is_available(),
                                                                   num_workers=processing_arguments.num_workers)

    processing_arguments.testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=setup_arguments.batch_sz,
                                                                  shuffle=False, pin_memory=torch.cuda.is_available(),
                                                                  num_workers=processing_arguments.num_workers)

    print("Data is ready")
