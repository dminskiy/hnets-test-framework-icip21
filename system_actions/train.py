import time

from .test import get_num_correct_pred

def train_epoch(processing_arguments):
    train_start = time.time()

    model = processing_arguments.model
    optimiser = processing_arguments.optimiser
    criterion = processing_arguments.criterion
    trainloader = processing_arguments.trainloader
    device = processing_arguments.shared_arguments.device
    use_scheduler = processing_arguments.shared_arguments.use_scheduler
    scheduler_type = processing_arguments.shared_arguments.scheduler_type

    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    loss = None
    lr = None

    total_num_images_for_training = int(len(trainloader.dataset)*processing_arguments.partion_tr_data + 0.5)
    images_used_sofar = 0

    for param_group in optimiser.param_groups:
        lr = param_group['lr']

    for batch_idx, (data, target) in enumerate(trainloader):
        if images_used_sofar >= total_num_images_for_training:
            break

        processing_arguments.optimiser.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()

        loss.backward()
        optimiser.step()

        correct += get_num_correct_pred(output, target, k=1)
        total += target.size(0)

        images_used_sofar += len(data)
        #TODO make dependant on the batch size. Get bs from trainloader
        if batch_idx % 10 == 0:
            print('Training: [{}/{} ({:.1f}%)]\tLoss: {:.4f}'.format(
                                                                    images_used_sofar,
                                                                    total_num_images_for_training,
                                                                    100. * images_used_sofar/ total_num_images_for_training, loss.item()))

    acc = 100. * correct / total
    print("correct: {}; total: {}; acc = {:.2f}; loss: {:.4f}".format(correct,total,acc,loss.item()))

    if use_scheduler == 1:
        #each epoch
        if scheduler_type == 'onPlateau':
            processing_arguments.scheduler.step(running_loss)
        else:
            processing_arguments.scheduler.step()

    train_end = time.time()

    train_time = train_end - train_start

    return acc, lr, train_time, loss