from .format_val import format_val
import wget
import os
import zipfile
import shutil

def prepare_tinyimagenet(root_dir):

    if not os.path.exists(root_dir):
        raise NotADirectoryError("Root directory doesn't exist: [{}]".format(root_dir))

    print("Preparing Tiny Imagenet dataset\nDownloading...")
    dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_filename = wget.download(dataset_url, out = root_dir)

    full_path_zip_file  = os.path.join(root_dir, zip_filename)

    print("Unziping...")
    with zipfile.ZipFile(full_path_zip_file, 'r') as zip_ref:
        zip_ref.extractall(root_dir)

    test_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'test')
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except OSError as e:
            raise(RuntimeError("Couldn't delete test directory. Error: %s : %s" % (test_dir, e.strerror)))
        except:
            raise (RuntimeError("Couldn't delete test directory: [{}]".format(test_dir)))

    print("Preparing data...")
    format_val(root_dir)

    print("Imagenet dataset is ready.")