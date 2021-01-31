import datetime
import os
import warnings

from openpyxl import Workbook

from results_decoder.utils.results_processing_utils import create_dir

class ReportData():
    def __init__(self):
        self.dataset = None
        self.data_partition = None
        self.data_size = None
        self.augment_data = None
        self.n_classes = None
        self.batch_sz = None
        self.scat_enabled = None
        self.scat_order = None
        self.scat_type = None
        self.J = None
        self.L = None
        self.classifier = None
        self.lr = None
        self.planned_epochs = None
        self.epochs = None
        self.total_tr_time = None
        self.total_tst_time = None
        self.final_accuracy = None
        self.top_5_accuracy = None
        self.train_accuracies = None
        self.test_accuracies = None
        self.test_epochs = None
        self.device = None
        self.num_devices = None
        self.optimiser = None
        self.scheduler_type = None
        self.scheduler_step = None
        self.dnn_dropout = None
        self.net_learnable_params = None
        self.scat_learnable_params = None
        self.lrs = None
        self.train_losses = None
        self.confusion_mat = None
        self.datetime_start = None
        self.datetime_end = None
        self.cost_func = None
        self.test_accuracies_top_5 = None

        self.total_execution_time = None
        self.total_execution_time_sec = None

        self.min_test_err = None
        self.min_test_err_epoch = None
        self.min_av_test_err = None
        self.last_error_av = None

        self.img_path_detailed_errs = None
        self.img_path_losses_n_errs = None
        self.img_path_lrs_n_errs = None
        self.img_path_confusion_mat = None

        self.keys_extracted = []

        self.data_isReady = False

    def sec2time(self, sec):
        return datetime.timedelta(seconds=sec).__str__().split(".")[0]

    def set_time(self):
        if self.is_available(self.datetime_end) and self.is_available(self.datetime_start):
            self.total_execution_time = self.datetime_end - self.datetime_start
            self.total_execution_time_sec = self.total_execution_time.total_seconds()
            self.total_execution_time = self.total_execution_time.__str__().split(".")[0]
        elif self.is_available(self.total_tr_time) and self.is_available(self.total_tst_time):
            self.total_execution_time = self.total_tst_time + self.total_tr_time
            self.total_execution_time_sec = self.total_execution_time
            self.total_execution_time = self.sec2time(self.total_execution_time)
        else:
            self.total_execution_time = None

    def str2datetime(self, str):
        if str is None or str == "N/A":
            return str

        str_split = str.split(";")
        if len(str_split) != 2:
            warnings.warn("Couldn't convert a string to datetime: [{}]. Returning the original string.".format(str), Warning)
            return str

        date = str_split[0]
        time = str_split[1]

        date_split = date.split("-")
        y = int(date_split[0])
        m = int(date_split[1])
        d = int(date_split[2])

        time_split = time.split(":")
        h = int(time_split[0])
        min = int(time_split[1])
        s = int(time_split[2])

        return datetime.datetime(y,m,d,h,min,s)

    def key_vefied(self, key, dict):
        if not key in dict:
            return False

        if key in self.keys_extracted:
            warnings.warn("Data at key: [{}] is being extracted again".format(key), Warning)

        return True

    def extract_data(self, key, dict):
        if not self.key_vefied(key, dict):
            return None

        val = dict[key]
        self.keys_extracted.append(key)

        if val is None or val == -1:
            return "N/A"
        else:
            if val == 1:
                return True
            elif val == 0:
                return False
            else:
                return val

    def extract_array_data(self, key, dict):
        if not self.key_vefied(key, dict):
            return None

        val = dict[key]
        self.keys_extracted.append(key)

        if val is None or val == -1:
            return []
        else:
            if isinstance(val, list):
                return val
            else:
                ValueError("Key [{}] doesn't contain an array".format(key))

    def is_enabled(self, key, dict):
        if not self.key_vefied(key, dict):
            return None

        val = dict[key]
        self.keys_extracted.append(key)

        if val == True or val == 1:
            return True
        else:
            return False

    def is_available(self, val):
        if val is not None and val != "N/A" and val != []:
            return True
        else:
            return False

    def load_report_data(self, input_data):

        self.confusion_mat = self.extract_data("confusion_mat", input_data)

        if self.is_enabled("use_scater", input_data):
            self.scat_enabled = True
            self.scat_order = self.extract_data("scat_order", input_data)
            self.scat_type = self.extract_data("scat_type", input_data)
        else:
            self.scat_enabled = False

        if self.is_enabled("use_scheduler", input_data):
            self.use_scheduler = True
            self.scheduler_type = self.extract_data("scheduler_type", input_data)
            self.scheduler_step = self.extract_data("scheduler_step", input_data)
        else:
            self.use_scheduler = False

        self.optimiser = self.extract_data("optimiser", input_data)

        self.train_accuracies = self.extract_array_data("train_accs", input_data)
        self.test_accuracies = self.extract_array_data("test_accs", input_data)
        self.test_epochs = self.extract_array_data("test_epochs", input_data)
        self.lrs = self.extract_array_data("lrs", input_data)
        self.train_losses = self.extract_array_data("train_losses", input_data)
        self.test_accuracies_top_5 = self.extract_array_data("test_accs_top5", input_data)

        self.dataset = self.extract_data("data", input_data)
        self.data_partition = self.extract_data("data_partition",input_data)
        self.data_size = self.extract_data("data_size",input_data)
        self.augment_data = self.extract_data("augment_data", input_data)
        self.n_classes = self.extract_data("n_classes", input_data)
        self.batch_sz = self.extract_data("batch_sz", input_data)
        self.final_accuracy = self.extract_data("final_acc", input_data)
        self.top_5_accuracy = self.extract_data("top_5", input_data)
        self.classifier = self.extract_data("classifier", input_data)
        self.epochs = self.extract_data("last_epoch", input_data)
        self.planned_epochs = self.extract_data("epochs", input_data)
        self.lr = self.extract_data("lr", input_data)
        self.J = self.extract_data("J", input_data)
        self.L = self.extract_data("L", input_data)
        self.total_tr_time = self.extract_data("total_tr_time", input_data)
        self.total_tst_time = self.extract_data("total_tst_time", input_data)
        self.device = self.extract_data("device", input_data)
        self.num_devices = self.extract_data("num_devices", input_data)
        self.dnn_dropout = self.extract_data("wrn_dropout", input_data)
        self.net_learnable_params = self.extract_data("net_num_params", input_data)
        self.scat_learnable_params = self.extract_data("scat_learn_params", input_data)
        self.cost_func = self.extract_data("cost_func", input_data)

        self.datetime_end = self.str2datetime(self.extract_data("datetime_end", input_data))
        self.datetime_start = self.str2datetime(self.extract_data("datetime_start", input_data))
        self.set_time()

        self.data_isReady = True

class ReportDataTable():
    def __init__(self):
        self.table_isReady = False
        self.headers = ["Full Path", "File Name", "Total Epochs", "Trained Epochs", "Dataset", "Data Used, %",
                        "Batch Size", "Input size", "Network", "ScatNet", "ScatNet Order", "Scat J",
                        "Top-1 last epoch, %", "Top-1 last epoch av, %", "Top-1 min, %", "Top-1 min av, %",
                        "Top-1 min, epoch", "Top-5 last epoch, %", "Total Execution time", "Training time per epoch, sec",
                        "Number of learnable params", "Data Augmentation", "Device Name(s)", "Optimiser",
                        "Scheduler Type", "Scheduler Step", "Starting LR"]
        self.path_out = None
        self.file_name = None
        self.workbook = None
        self.spreadsheet = None
        self.next_row = 1
        self.first_row = 1
        self.first_col = 1

    def initialise(self, path_out, file_name):
        #create file
        self.path_out = path_out

        if not create_dir(self.path_out):
            warnings.warn("Directory existed already [{}]. Beware of file overwriting".format(self.path_out))

        if not file_name.endswith(".xlsx"):
            self.file_name = file_name + ".xlsx"

        self.workbook = Workbook()
        self.spreadsheet = self.workbook.active
        self.spreadsheet.title = "Results"

        #init headers
        self.add_row(arr=self.headers, row=self.first_row, starting_col=self.first_col)

        self.next_row += 1
        self.table_isReady = True

    def update(self, report_data, json_dir, json_name):
        #add report data as per header
        if self.table_isReady:
            #TODO check times. Can we have tr time from data?
            traning_time_per_epoch = report_data.total_tr_time / report_data.epochs
            vals = [json_dir, json_name, report_data.planned_epochs, report_data.epochs, report_data.dataset,
                    report_data.data_partition, report_data.batch_sz, report_data.data_size, report_data.classifier,
                    report_data.scat_type, report_data.scat_order, report_data.J, report_data.final_accuracy,
                    100.-report_data.last_error_av, 100.-report_data.min_test_err, 100.-report_data.min_av_test_err,
                    report_data.min_test_err_epoch, report_data.top_5_accuracy, report_data.total_execution_time,
                    traning_time_per_epoch, report_data.net_learnable_params, report_data.augment_data,
                    report_data.device, report_data.optimiser, report_data.scheduler_type, report_data.scheduler_step,
                    report_data.lr]

            self.add_row(arr=vals, row=self.next_row, starting_col=self.first_col)
            self.next_row += 1
        else:
            warnings.warn("Cannot update table. The table is not ready")

    def add_row(self, arr, row, starting_col):
        for col in range(0, len(arr)):
            self.spreadsheet.cell(row=row, column=starting_col+col).value = arr[col]
        pass

    def save(self):
        self.workbook.save(os.path.join(self.path_out, self.file_name))
        self.close()

    def close(self):
        #close the file
        self.workbook.close()
        pass