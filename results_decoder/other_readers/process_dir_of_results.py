import argparse
import sys
import time
import datetime

from results_decoder.utils.ReportDataHolder import ReportData, ReportDataTable
from results_decoder.other_readers.report_from_json import generate_and_save_report

from results_decoder.utils.results_processing_utils import *

STD_RESULTS_DIR_NAME = "0.RESULTS_COMBINED"
STD_REPORTS_DIR = os.path.join(STD_RESULTS_DIR_NAME, "REPORTS")
STD_MISSING_FILES_DIR = os.path.join(STD_RESULTS_DIR_NAME, "MISSING")
STD_TABLE_DIR = os.path.join(STD_RESULTS_DIR_NAME, "TABLE")

FINAL_MODEL_STD_ENDING = ".MODEL_FINAL.torchdict"
STD_MODEL_CHPT_ENDING = ".torchdict"
FINAL_JSON_STD_ENDING = ".DATA_FINAL.json"
STD_DATA_CHECKPT_DIR = os.path.join("CHECKPTS", "data")
STD_MODEL_CHECKPT_DIR = os.path.join("CHECKPTS", "model")

FINAL_JSON_INFO_TMPL = {
    "dir":"",
    "filename":""
}

def parse_input_arguments(input):
    parser = argparse.ArgumentParser()

    parser.add_argument('-results_dir', default=None, nargs="?", type=str, required=False,
                        help="Directory that contains subdirectories with standard file structure. "
                             "Expected: 'filename'.DATA_FINAL.json, 'filename'.MODEL_FINAL.torchdict and CHECKPTS subdir")
    parser.add_argument('-results_top_dir', default=None, nargs="?", type=str, required=False,
                        help="Directory that contains results directories as sub-directories. ")
    parser.add_argument("-out_dir", default=None, nargs="?", type=str, required=False,
                        help="Directory where the results will be saved. If not specified the results will be saved in the -results_dir")
    parser.add_argument("-exclude_dirs", default=None, nargs="*", required=False, help="Specify directory names to be "
                                                                                       "exlcuded when looking for "
                                                                                       "results directories. Should be used"
                                                                                       "with -results_top_dir to have an effect")

    args = parser.parse_args(input)

    if args.results_dir is None and args.results_top_dir is None:
        raise (ValueError("Please specify either -results_dir or -results_top_dir"))

    return args


def setup_out_dirs(all_reports, table, missing_files):

    if not os.path.isdir(out_dir):
        raise (FileExistsError("Cannot write output because output directory doesn't exist: [{}]".format(out_dir)))

    if not create_dir(all_reports):
        warnings.warn("File existed already. May overwrite some files in dir: [{}]".format(all_reports))
    if not create_dir(table):
        warnings.warn("File existed already. May overwrite some files in dir: [{}]".format(table))
    if not create_dir(missing_files):
        warnings.warn("File existed already. May overwrite some files in dir: [{}]".format(missing_files))


def get_report_data(dir, missing_files):

    final_model_file = get_file(dir, FINAL_MODEL_STD_ENDING)
    final_json_file = get_file(dir, FINAL_JSON_STD_ENDING)

    final_json_info = FINAL_JSON_INFO_TMPL.copy()
    final_json_info["filename"] = final_json_file
    final_json_info["dir"] = dir

    if final_model_file is None:
        model_checkpoints_dir = os.path.join(dir, STD_MODEL_CHECKPT_DIR)
        if is_empty_dir(model_checkpoints_dir) and get_file(dir, STD_MODEL_CHPT_ENDING) is None:
            missing_files["empty_model"].append(dir)
        else:
            missing_files["missing_final_model"].append(dir)

    if final_json_file is None:
        data_checkpoints_dir = os.path.join(dir, STD_DATA_CHECKPT_DIR)
        data_checkpoint = get_file(data_checkpoints_dir, ".json")
        if data_checkpoint is None:
            missing_files["empty_data"].append(dir)
        else:
            missing_files["missing_final_data"].append(dir)
            final_json_info["filename"] = data_checkpoint
            final_json_info["dir"] = data_checkpoints_dir

    report_data = None
    if final_json_info["filename"] is not None:
        json_path = os.path.join(final_json_info["dir"],final_json_info["filename"])
        all_data = load_json(json_path)
        report_data = ReportData()
        report_data.load_report_data(all_data)

    return report_data, final_json_info


def process_directory(dir, out_dir, report_data_table):
    all_reports_dir = os.path.join(out_dir,STD_REPORTS_DIR)
    table_dir = os.path.join(out_dir, STD_TABLE_DIR)
    missing_files_dir = os.path.join(out_dir, STD_MISSING_FILES_DIR)
    setup_out_dirs(missing_files=missing_files_dir, table=table_dir, all_reports=all_reports_dir)

    missing_files = {
        "empty_data": [],
        "empty_model": [],
        "missing_final_data": [],
        "missing_final_model": []
    }

    table_name = dir.split("/")
    table_name = '_'.join(table_name[len(table_name)-3:])

    report_data_table.initialise(file_name=table_name, path_out=table_dir)

    # results folder contains outputs for individual experiment
    total_folders = 0
    for results_folder in os.listdir(dir):
        full_path_to_results_folder = os.path.join(dir, results_folder)
        if os.path.isdir(full_path_to_results_folder) and results_folder != STD_RESULTS_DIR_NAME:
            total_folders += 1
            report_data, final_json_info = get_report_data(full_path_to_results_folder, missing_files)

            if report_data is not None:
                generate_and_save_report(report_data=report_data, path_in=final_json_info["dir"],
                                         path_out=all_reports_dir, file_in=final_json_info["filename"], show_figs=False, save_pdf=False)
                report_data_table.update(report_data=report_data, json_dir=final_json_info["dir"], json_name=final_json_info["filename"])


    for key in missing_files:
        array_to_txt(array=missing_files[key], txt_full_path=os.path.join(missing_files_dir, key+".txt"))

if __name__ == '__main__':
    start = time.time()

    args = parse_input_arguments(sys.argv[1:])

    dirs = []
    out_dir = None

    if args.results_top_dir is not None:
        for item in os.listdir(args.results_top_dir):
            item_full_path = os.path.join(args.results_top_dir, item)
            if os.path.isdir(item_full_path):
                if args.exclude_dirs is not None:
                    if not (item in args.exclude_dirs):
                        dirs.append(os.path.join(args.results_top_dir, item))
                else:
                    dirs.append(os.path.join(args.results_top_dir, item))
    elif dir_does_exist(args.results_dir):
        dirs.append(args.results_dir)
        if args.out_dir is not None:
            out_dir = args.out_dir
    else:
        raise (FileExistsError("Results directory doesn't exist: [{}]".format(args.results_dir)))

    if is_empty_array(dirs):
        raise (FileExistsError("Please check the -results_top_dir. No sub-directories were found: [{}]".format(args.results_top_dir)))

    #dir here contains folders with results
    for dir in dirs:
        if args.out_dir is None or len(dirs) > 1:
            out_dir = dir
        report_data_table = ReportDataTable()
        process_directory(dir, out_dir, report_data_table)
        report_data_table.save()

    end = time.time()
    total_time = end - start
    total_time_processed = datetime.timedelta(seconds=total_time)

    print("Directories processed:")
    for dir in dirs:
        print(dir)

    print("\nExecution time: {}\nSeconds: {}".format(total_time_processed, total_time))