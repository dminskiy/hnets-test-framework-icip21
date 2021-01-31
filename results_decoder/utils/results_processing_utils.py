import os
import warnings
import json

def get_file(dir, file_ending):
    try:
        for file in os.listdir(dir):
            if file.endswith(file_ending):
                return file
        else:
            return None
    except:
        return None

def is_empty_array(array):
    return len(array) == 0


def is_empty_dir(dir):
    return len(os.listdir(dir)) == 0


def dir_does_exist(dir):
    if not os.path.isdir(dir):
        return False
    return True


def array_to_txt(array, txt_full_path):
    try:
        with open(txt_full_path, 'w') as f:
            for item in array:
                f.write("\n{}".format(item))
    except OSError as e:
        warnings.warn("Couldn't save a txt file: [{}]\nError: [{}]".format(txt_full_path, e))


def create_dir(dir):
    if not os.path.exists(dir):
        try:
            print("Path didn't exist. Trying to make dir: [{}]".format(dir))
            os.makedirs(dir)
            print("Directory created successfully.")
        except:
            print("Couldn't create saving dir: [{}]\nThe file was not saved.".format(dir), Warning)
            return True
    return False


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        try:
            print("Path didn't exist. Trying to make dir: [{}]".format(dir))
            os.mkdir(dir)
            print("Directory created successfully.")
            return
        except:
            print("Couldn't create saving dir: [{}]\nThe file was not saved.".format(dir), Warning)
            return


def load_json(filename):
    if not filename.endswith(".json"):
        raise(ValueError("Input file name extension is not json: {}".format(filename)))
    if not os.path.exists(filename):
        raise(FileNotFoundError("Input file was not found: {}".format(filename)))

    with open(filename, 'r') as file:
        json_data = json.load(file)

    return json_data