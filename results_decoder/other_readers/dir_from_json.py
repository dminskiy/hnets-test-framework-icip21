import os
import argparse
import sys

if __name__ == '__main__':

    full_decoder_path = '/home/dmitryminskiy/PycharmProjects/hybrids-test-framework/results_decoder/report_from_json.py'

    parser = argparse.ArgumentParser(description='Processing of the ScatNet results in JSON')

    parser.add_argument('-dir_in', nargs='?', type=str, required=True, help="Directory in which the result files are stored")
    parser.add_argument('-dir_out', nargs='?', type=str, required=True, help="Output directory")

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.dir_in):
        raise("The specified input directory doesnt exits: {}".format(args.dir_in))

    if not os.path.exists(args.dir_out):
        raise("The specified output directory doesnt exits: {}".format(args.dir_out))

    for file in os.listdir(args.dir_in):
        if file.endswith(".json"):
            json2process = file
            command = "python " + full_decoder_path +" -file_in " + json2process + " -path_in " + args.dir_in + " -path_out " + args.dir_out + " -show_figs 0 -save_pdf 0"
            os.system(command)