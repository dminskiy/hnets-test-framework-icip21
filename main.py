from system_actions.train import train_epoch
from system_actions.test import test_epoch, final_evaluation
from system_init.data_arguments_manager import SharedArguments, OutputArguments, SetupArguments, ProcessingAgruments
from system_init.image_data_manager import prepare_image_data
from system_init.input_manager import *
from system_init.model_manager import build_model

import sys, os
import datetime

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    datetime_start = datetime.datetime.now()

    shared_argumets = SharedArguments()
    setup_arguments = SetupArguments(shared_argumets)            #Should be separate from processing arguments
    processing_arguments = ProcessingAgruments(shared_argumets)  #Should be separate from setup arguments
    out_arguments = OutputArguments(shared_argumets)

    arguments_container = {
                            'setup_args': setup_arguments,
                            'processing_args': processing_arguments,
                            'shared_args': shared_argumets,
                            'out_args': out_arguments
                           }


    in_args = parse_input(sys.argv[1:])
    fill_arguments_container(in_args, arguments_container)

    setup_arguments.set_env(sys.argv[1:])

    #takes the input params and sets train and test loaders in processing_arguments
    prepare_image_data(setup_arguments, processing_arguments)

    build_model(processing_arguments, setup_arguments)

    out_arguments.calculate_model_coefs(processing_arguments.model.parameters())

    if out_arguments.print_net_layout == True:
        print(processing_arguments.model)

    #Fix the summary. Issues with calculation of parameters + scatterNet summary is missing
    out_arguments.print_pretraining_summary(processing_arguments.model.parameters(), setup_arguments)

    test_time = 0
    test_acc = -1
    last_test_epoch = -1
    train_loss = None
    global_epoch = -1

    try:
        for epoch in range(1, 1 + setup_arguments.epochs):
            global_epoch = epoch
            print("\n********** Staring Epoch {}".format(epoch))
            #training
            train_acc, lr, train_time, train_loss = train_epoch(processing_arguments)
            out_arguments.epoch_update_train(train_time, train_acc, lr, epoch, train_loss.item())

            #testing
            if out_arguments.do_checkpoint:
                #evaluate model and save data checkpoint
                if epoch % out_arguments.data_checkpoint_freq == 0 and out_arguments.data_checkpoint_freq is not -1:
                    checkpoint_name = shared_argumets.json_filename + ".DATA_CHECKPT"
                    data_checkpoint_full_path = os.path.join(shared_argumets.checkpoint_dir, "data", checkpoint_name) \
                            if shared_argumets.use_checkpoint_subdirs else os.path.join(shared_argumets.out_dir, checkpoint_name)
                    #test model
                    last_test_epoch = epoch
                    test_acc, test_acc_top_5, test_time = test_epoch(processing_arguments)
                    out_arguments.epoch_update_test(test_time, test_acc, test_acc_top_5, last_test_epoch)

                    #save info json
                    out_arguments.save_info_to_json(final_acc=test_acc, top_5=test_acc_top_5, setup_args=setup_arguments, file_name=data_checkpoint_full_path, data_partition=processing_arguments.partion_tr_data)

                #save model
                if epoch % out_arguments.model_checkpoint_freq == 0 and out_arguments.model_checkpoint_freq is not -1:
                    checkpoint_name = shared_argumets.std_filename + ".MODEL_CHECKPT_AT_EPOCH_" + str(epoch) + ".torchdict"
                    model_checkpoint_full_path = os.path.join(shared_argumets.checkpoint_dir, "model", checkpoint_name) \
                        if shared_argumets.use_checkpoint_subdirs else os.path.join(shared_argumets.out_dir, checkpoint_name)

                    out_arguments.save_model_params(processing_arguments, epoch, train_loss, model_checkpoint_full_path)

            out_arguments.print_epoch_summary()

        print("End of Training.")

    except (KeyboardInterrupt, SystemExit):
        if out_arguments.save_final_model:
            interrupt_model_full_path = os.path.join(shared_argumets.out_dir,
                                                 shared_argumets.std_filename + ".MODEL_INTERRUPTED_AT_EPOCH_" + str(global_epoch) + ".torchdict")
            out_arguments.save_model_params(processing_arguments, setup_arguments.epochs, train_loss, interrupt_model_full_path)
        try:
            sys.exit(-1)
        except SystemExit:
            os._exit(-1)

    datetime_end = datetime.datetime.now()

    out_arguments.date_time_start = datetime_start
    out_arguments.date_time_end = datetime_end

    if out_arguments.save_final_model:
        print("Saving final model parameters...")
        final_model_full_path = os.path.join(shared_argumets.out_dir, shared_argumets.std_filename + ".MODEL_FINAL.torchdict")
        out_arguments.save_model_params(processing_arguments, setup_arguments.epochs, train_loss, final_model_full_path)
        print("Parameters saved!")

    print("Evaluating the network now...")
    # final test
    final_acc, top_5, confusion_matrix = final_evaluation(processing_arguments)

    final_data_full_path = os.path.join(shared_argumets.out_dir, shared_argumets.json_filename + ".DATA_FINAL.json")
    out_arguments.save_info_to_json(final_acc=final_acc, top_5=top_5, setup_args=setup_arguments, file_name=final_data_full_path, data_partition=processing_arguments.partion_tr_data, confusion_matrix=confusion_matrix)
    out_arguments.print_training_summary(final_acc, top_5, confusion_matrix)

