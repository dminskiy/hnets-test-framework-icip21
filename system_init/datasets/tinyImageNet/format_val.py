import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

def format_val(root_dir = None):

    if root_dir == None:
        root_dir = '/user/HS221/dm00314/Desktop/TinyImageNet/'

    target_folder = join(root_dir,'tiny-imagenet-200/val/')
    test_folder   = join(root_dir,'tiny-imagenet-200/test/')

    annotations_txt = join(root_dir,'tiny-imagenet-200/val/val_annotations.txt')

    all_val_images_dir = join(root_dir,'tiny-imagenet-200/val/images/')
    all_val_images = join(root_dir,'tiny-imagenet-200/val/images/*')

    os.mkdir(test_folder)
    val_dict = {}
    with open(annotations_txt, 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(all_val_images)
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
        if not os.path.exists(test_folder + str(folder)):
            os.mkdir(test_folder + str(folder))
            os.mkdir(test_folder + str(folder) + '/images')


    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if len(glob.glob(target_folder + str(folder) + '/images/*')) <25:
            dest = target_folder + str(folder) + '/images/' + str(file)
        else:
            dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir(all_val_images_dir)
