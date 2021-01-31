import torch
import sys
import json
import numpy as np
import os
import datetime
import warnings

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SetupArguments:
    def __init__(self, shared_arguments):
        self.shared_arguments = shared_arguments
        self.cost_func = "CrossEntropyLoss" #TODO: change if more cost funcs are used

        self.J = None
        self.n_classes = None
        self.in_shape = None
        self.batch_sz = None
        self.classifier = None
        self.channels_after_scaternet = None
        self.in_channels = None
        self.lr = None
        self.epochs = None
        self.scat_order = None
        self.optim = None
        self.augment_data = None
        self.scheduler_step = None
        self.runon = None
        self.set_datasize = None
        self.data_root = None
        self.dataset = None
        self.num_workers = None
        self.half_scat_feat_resolution = None

    def dir_exists(self, dir):
        if not os.path.exists(dir):
            try:
                print("Path didn't exist. Trying to make dir: [{}]".format(dir))
                os.makedirs(dir)
                print("Directory created successfully.")
            except OSError as e:
                warnings.warn("Couldn't create saving dir: [{}]\nThe file was not saved.\nError: {}".format(dir, e))
                return False
        return True

    def set_env(self, sys_args):
        #make sure out dir exists, make otherwise
        if not self.dir_exists(self.shared_arguments.out_dir): #TODO use later in code
            raise(NotADirectoryError("Output directory doesn't exist and couldn't be created: {}".format(self.shared_arguments.out_dir)))

        #make sure checkpoints dir exists, if so create /model and /data dirs
        self.shared_arguments.checkpoint_dir = os.path.join(self.shared_arguments.out_dir, "CHECKPTS") if  self.shared_arguments.checkpoint_dir == None else self.shared_arguments.checkpoint_dir

        if not self.dir_exists(self.shared_arguments.checkpoint_dir):
            raise(NotADirectoryError("Output directory doesn't exist and couldn't be created: {}".format(self.shared_arguments.out_dir)))

        if self.dir_exists(os.path.join(self.shared_arguments.checkpoint_dir, "model")) and self.dir_exists(os.path.join(self.shared_arguments.checkpoint_dir, "data")):
            self.shared_arguments.use_checkpoint_subdirs = True
        else:
            self.shared_arguments.use_checkpoint_subdirs = False

        #If no json name specified, set to std filename, strip .json
        if self.shared_arguments.json_filename == None:
            self.shared_arguments.json_filename = self.shared_arguments.std_filename
        else:
            if self.shared_arguments.json_filename.endswith('.json'):
                self.shared_arguments.json_filename.replace('.json', '')

        #Save input sys arguments to a txt file
        args_out_file_path = os.path.join(self.shared_arguments.out_dir, self.shared_arguments.std_filename + ".SYS_ARGS_INPUT.txt")
        args_out_file = open(args_out_file_path, "w")
        for x in sys_args:
            args_out_file.write(x + " ")
        args_out_file.close()

class ProcessingAgruments:
    def __init__(self, shared_arguments):
        self.shared_arguments = shared_arguments
        self.fix_seeds = None
        self.num_workers = None
        self.trainloader = None
        self.testloader = None
        self.partion_tr_data = None

        self.scatNet = None
        self.model = None
        self.optimiser = None
        self.scheduler = None
        self.criterion = None

    def is_mallat_scatter(self):
        return self.shared_arguments.enable_scat == 1 and self.shared_arguments.scat_type == 'mallat'
    def is_on_gpu(self):
        return torch.cuda.is_available() and self.shared_arguments.explicit_cpu == 0

class SharedArguments:
    def __init__(self):
        self.device = None #TODO get a list, save as string
        self.enable_scat = None
        self.scat_type = None
        self.explicit_cpu = None
        self.use_scheduler = None
        self.scheduler_type = None
        self.classes = None
        self.num_devices_used = -1

        self.std_filename = None
        self.out_dir = None
        self.checkpoint_dir = None
        self.use_checkpoint_subdirs = False
        self.json_filename = None
        self.device_name = "N/A"

class OutputArguments():
    def __init__(self, shared_argumets):
        self.shared_arguments = shared_argumets

        self.print_net_layout = True
        self.print_num_of_net_coefs = True

        self.total_train_time = -1
        self.total_test_time = -1

        self.train_accuracies = []
        self.test_accuracies = []
        self.test_accuracies_top_5 = []
        self.train_lrs = []
        self.all_train_epochs = []
        self.all_test_epochs = []
        self.training_loss = []

        self.last_epoch_train_time = -1
        self.last_epoch_test_time = -1
        self.last_epoch_train_accuracy = -1
        self.last_epoch_test_accuracy = -1
        self.last_epoch = -1
        self.last_test_epoch = -1
        self.last_epoch_test_accuracy_top_5 = -1

        self.confusion_matrix = None

        self.net_learnable_params = -1
        self.scat_learnable_params= -1

        self.do_checkpoint = False
        self.model_checkpoint_freq = -1
        self.data_checkpoint_freq = -1
        self.save_final_model = False

        self.date_time_start = None
        self.date_time_end = None

    def datetime2str(self, date_time):
        if not isinstance(date_time, datetime.datetime):
            return date_time
        return "{}-{}-{};{}:{}:{}".format(date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second)

    def print_pretraining_summary(self, model_params, setup_args):
        device_name = self.shared_arguments.device_name

        print("Dataset: {}".format(setup_args.dataset))
        print("Batch size: {}".format(setup_args.batch_sz))
        print("Epochs: {}".format(setup_args.epochs))

        print("CNN: {}".format(setup_args.classifier))

        print("\nScattering is enabled: {}".format(self.shared_arguments.enable_scat))
        print("Scatterig type: {}".format(self.shared_arguments.scat_type))
        print("Scatterig order: {}".format(setup_args.scat_order))
        print("Scatterig J: {}".format(setup_args.J))


        if self.print_num_of_net_coefs:
            learnable_net_coefs, total_net_coefs = self.calculate_model_coefs(model_params)
            print("###### The Network Summary:")
            print("###### Total number of parameters: {}"
                  "\n###### Learnable parameters      : {}".format(total_net_coefs, learnable_net_coefs))

        print("Tests are performed on {} using {} devices".format(device_name, self.shared_arguments.num_devices_used))

    def print_epoch_summary(self):
        print("************************ EPOCH: {} ************************".format(self.last_epoch))
        print("###### Training time        : {:.2f}".format(self.last_epoch_train_time))
        print("###### Training Error, %    : {:.2f}".format(100 - self.last_epoch_train_accuracy))
        if self.last_epoch == self.last_test_epoch:
            print("###### Testing time           : {:.2f}".format(self.last_epoch_test_time))
            print("###### Testing Error, %       : {:.2f}".format(100 - self.last_epoch_test_accuracy))
            print("###### Testing Error Top 5, % : {:.2f}".format(100 - self.last_epoch_test_accuracy_top_5))

    def print_training_summary(self, final_accuracy, final_accuracy_top_5, confusion_matrix):
        print("!!!!! Backup. Errors and LR: !!!!!")
        self.print_backup_array("Y_trainErr", self.train_accuracies)
        self.print_backup_array("Y_testErr", self.test_accuracies)
        self.print_backup_array("Learning Rate", self.train_lrs, False)

        #self.print_pretraining_summary() TODO sort the pretraining summary
        print()
        print("\n************************ TOTAL EPOCHS: {} ************************".format(self.last_epoch))
        print("###### Total Training time        : {:.2f}".format(self.total_train_time))
        print("###### Final Error, %             : {:.2f}".format(100 - final_accuracy))
        print("###### Final Error Top-5, %             : {:.2f}".format(100 - final_accuracy_top_5))

    def save_info_to_json(self, final_acc, top_5, setup_args, file_name, data_partition, confusion_matrix = None):
        device_name = self.shared_arguments.device_name

        out_info = {}
        if confusion_matrix is not None:
            confusion_matrix = np.array(confusion_matrix)
            out_info ['confusion_mat'] = confusion_matrix #use .reshape(n_classes,n_classes) to get to confusion matrix
        else:
            out_info['confusion_mat'] = None

        out_info ['data'] = setup_args.dataset
        out_info ['data_partition'] = int(data_partition*100.)
        out_info ['augment_data'] = setup_args.augment_data
        out_info ['n_classes'] = setup_args.n_classes
        out_info ['batch_sz'] = setup_args.batch_sz
        out_info ['use_scater'] = self.shared_arguments.enable_scat
        out_info ['scat_order'] = setup_args.scat_order
        out_info ['scat_type'] = self.shared_arguments.scat_type
        out_info ['classifier'] = setup_args.classifier
        out_info ['last_epoch'] = self.last_epoch
        out_info ['epochs'] = setup_args.epochs
        out_info ['lr'] = setup_args.lr
        out_info ['J'] = setup_args.J
        out_info ['L'] = 8
        out_info ['total_tr_time'] = self.total_train_time
        out_info ['total_tst_time'] = self.total_test_time
        out_info ['final_acc'] = final_acc
        out_info ['top_5'] = top_5
        out_info ['train_accs'] = self.train_accuracies
        out_info ['test_epochs'] = self.all_test_epochs
        out_info ['test_accs'] = self.test_accuracies
        out_info ['device'] = device_name
        out_info ['num_devices'] = self.shared_arguments.num_devices_used
        out_info ['optimiser'] = setup_args.optim
        out_info ['use_scheduler'] = self.shared_arguments.use_scheduler
        out_info ['scheduler_type'] = self.shared_arguments.scheduler_type
        out_info ['scheduler_step'] = setup_args.scheduler_step
        out_info ['augment_data'] = setup_args.augment_data
        out_info ['wrn_dropout'] = setup_args.wrn_dropout
        out_info ['net_num_params'] = self.net_learnable_params
        out_info ['scat_learn_params'] = self.scat_learnable_params
        out_info ['lrs'] = self.train_lrs
        out_info ['train_losses'] = self.training_loss
        out_info ['datetime_start'] = self.datetime2str(self.date_time_start)
        out_info ['datetime_end'] = self.datetime2str(self.date_time_end)
        out_info ['cost_func'] = setup_args.cost_func
        out_info ['test_accs_top5'] = self.test_accuracies_top_5
        out_info ['data_size'] = setup_args.in_shape

        out_json = json.dumps(out_info, cls=NumpyEncoder)

        print("!!!!! Backup. JSON: ")
        print(out_json)

        if not file_name.endswith('.json'):
            file_name += ".json"

        try:
            with open(file_name, 'w') as outfile:
                json.dump(out_info, outfile, cls=NumpyEncoder)
                outfile.close()
        except IOError:
            outfile = file_name
            self.ensure_dir_created(outfile)
            f = open(outfile, "w")
            json.dump(out_info,f,cls=NumpyEncoder)
            outfile.close()

    def epoch_update_train(self,train_time, train_acc, lr, epoch,loss):
        self.last_epoch_train_time = train_time
        self.last_epoch_train_accuracy = train_acc
        self.train_accuracies.append(train_acc)
        self.train_lrs.append(lr)
        self.last_epoch = epoch
        self.training_loss.append(loss)
        self.total_train_time += train_time
        self.all_train_epochs.append(epoch)

    def epoch_update_test(self, test_time, test_acc, test_acc_top_5, last_test_epoch):
        self.last_test_epoch = last_test_epoch
        self.all_test_epochs.append(last_test_epoch)
        self.last_epoch_test_time = test_time
        self.total_test_time += test_time

        self.last_epoch_test_accuracy_top_5 = test_acc_top_5
        self.test_accuracies_top_5.append(test_acc_top_5)

        self.last_epoch_test_accuracy = test_acc
        self.test_accuracies.append(test_acc)

    def print_backup_array(self, name, data, is_acc=True):
        sys.stdout.write(name)
        sys.stdout.write(" = [")
        sys.stdout.flush()
        for i in range(len(data)):
            value = 100 - data[i] if is_acc else data[i]

            sys.stdout.write("%f" % (value)) if i == len(data) - 1 else sys.stdout.write("%f, " % (value))
            sys.stdout.flush()
        print("]")

    def calculate_model_coefs(self, model_params):
        learnable_net_coefs = sum(p.numel() for p in model_params if p.requires_grad)
        total_net_coefs = sum(p.numel() for p in model_params)

        return learnable_net_coefs, total_net_coefs

    def ensure_dir_created(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_model_params(self, processing_args, epoch, loss, full_file_name):
        saving_dict = {
            "epoch" : epoch,
            "model_state_dict": None,
            "optimiser_state_dict": processing_args.optimiser.state_dict(),
            "loss": loss
        }
        if self.shared_arguments.num_devices_used > 1:
            saving_dict["model_state_dict"] = processing_args.model.module.state_dict()
        else:
            saving_dict["model_state_dict"] = processing_args.model.state_dict()

        torch.save(saving_dict, full_file_name)