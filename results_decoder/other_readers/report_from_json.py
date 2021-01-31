import argparse
import time
from matplotlib import pyplot as plt
from docx2pdf import convert as docx2pdf_converter

from results_decoder.utils.ReportDataHolder import ReportData
from results_decoder.utils.report_generator_utils import *

FIGURES_FOLDER = "figures"
detailed_errors_prefix = "DetailedErrors."
losses_and_errors_prefix = "LossesAndErrors."
lrs_and_errors_prefix = "LrsAndErrors."
confusion_mat_prefix = "ConfusionMatrix."

def parse_input_arguments(input):
    parser = argparse.ArgumentParser(description='Processing of the ScatNet results in JSON')

    parser.add_argument('-file_in', nargs="?", type=str, required=True, help="The name of the json file with the results")
    parser.add_argument('-path_in', nargs="?", type=str, required=True, help="Path that contains the input file")
    parser.add_argument('-path_out', nargs="?", type=str, required=True, help="Output directory")
    parser.add_argument('-show_figs', nargs="?", type=int, choices=[0,1], required=True, help="0 or 1 to disable/enable showing of the figures. They will be saved anyway.")
    parser.add_argument('-save_pdf', nargs="?", type=int, choices=[0, 1], required=False, default=0,help="0 or 1 to disable/enable saving an additional pdf file.")

    args = parser.parse_args(input)

    args.show_figs = True if args.show_figs == 1 else False
    args.save_pdf = True if args.save_pdf == 1 else False

    return args

def generate_and_save_report(report_data, path_in, path_out, file_in, show_figs, save_pdf):

    # Create templates for graphs
    fig1 = plt.figure(figsize=(12, 10), dpi=100)
    fig2 = plt.figure(figsize=(12, 10), dpi=100)
    fig3 = plt.figure(figsize=(12, 10), dpi=100)

    plot_cm = True if report_data.is_available(report_data.confusion_mat) else False

    if report_data.n_classes < 25:
        fig4 = plt.figure(figsize=(12, 10), dpi=100)
    elif report_data.n_classes >= 25 and report_data.n_classes < 75:
        fig4 = plt.figure(figsize=(60, 50), dpi=50)
    elif report_data.n_classes >= 75 and report_data.n_classes < 110:
        fig4 = plt.figure(figsize=(90, 80), dpi=50)
    else:
        plot_cm = False

    # Generate Graphs
    plot_errors_with_stats(fig1, report_data)
    plot_errors_with_data(report_data, report_data.train_losses, fig2, title="Train Losses")
    plot_errors_with_data(report_data, report_data.lrs, fig3, title="Train Learning Rates")

    if plot_cm:
        plot_confusion_matrix(fig4, report_data.confusion_mat)

    # Save Graphs
    out_dir_suffix = file_in.strip(".json")
    path_out = os.path.join(path_out, out_dir_suffix)

    detailed_errors_name = detailed_errors_prefix + file_in + ".jpg"
    losses_and_errors_name = losses_and_errors_prefix + file_in + ".jpg"
    lrs_and_errors_name = lrs_and_errors_prefix + file_in + ".jpg"
    confusion_mat_name = confusion_mat_prefix + file_in + ".jpg"

    try:
        save_figure(fig1, path_out, detailed_errors_name, figures_folder=FIGURES_FOLDER)
        report_data.img_path_detailed_errs = os.path.join(path_out, FIGURES_FOLDER, detailed_errors_name)

        save_figure(fig2, path_out, losses_and_errors_name, figures_folder=FIGURES_FOLDER)
        report_data.img_path_losses_n_errs = os.path.join(path_out, FIGURES_FOLDER, losses_and_errors_name)

        save_figure(fig3, path_out, lrs_and_errors_name, figures_folder=FIGURES_FOLDER)
        report_data.img_path_lrs_n_errs = os.path.join(path_out, FIGURES_FOLDER, lrs_and_errors_name)

        if plot_cm:
            save_figure(fig4, path_out, confusion_mat_name, figures_folder=FIGURES_FOLDER)
            report_data.img_path_confusion_mat = os.path.join(path_out, FIGURES_FOLDER, confusion_mat_name)
    except:
        warnings.warn("Couldn't save the figures, for file: {}".format(os.path.join(path_in, file_in)))
        if show_figs:
            show_figs = False
            fig1.show()
            fig2.show()
            fig3.show()
            fig4.show()

    document_name = "ResultsSummary." + file_in + ".docx"
    full_document_name = os.path.join(path_out, document_name)

    try:
        generate_report_docx(report_data, path_out, document_name, file_in)
        if save_pdf:
            try:
                pdf_name = full_document_name.strip(".docx") + ".pdf"
                docx2pdf_converter(full_document_name, pdf_name)
            except:
                warnings.warn("Error while saving the report as pdf: [{}]".format(full_document_name), Warning)
                pass
    except:
        warnings.warn("Error while saving the report document: [{}]".format(full_document_name), Warning)
        pass

    # Show Graphs
    if show_figs:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

if __name__ == '__main__':
    proc_start = time.time()
    args = parse_input_arguments(sys.argv[1:])
    json_path = os.path.join(args.path_in, args.file_in)

    all_data = load_json(json_path)

    report_data = ReportData()
    report_data.load_report_data(all_data)

    generate_and_save_report(report_data=report_data, path_in=args.path_in, path_out=args.path_out,
                             file_in=args.file_in, show_figs=args.show_figs, save_pdf=args.save_pdf)

    print("Input file: {}\nEnd of processing. Time taken: {} sec"
          "\n*******************************************************\n".format(args.file_in, round(time.time() - proc_start)))