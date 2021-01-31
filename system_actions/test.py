from sklearn.metrics import confusion_matrix

def get_num_correct_pred(output, target, k=1):

    if k==1:
        pred = torch.argmax(output, 1)
        return pred.eq(target.view_as(pred)).sum().item()

    _, predk = output.topk(k)
    predk = predk.t()
    correct = predk.eq(target.view(1, -1).expand_as(predk))
    correct = correct.t()
    correct = correct.any(1).sum().item()

    return correct


def test_epoch(processing_arguments):
    test_start = time.time()

    model = processing_arguments.model
    criterion = processing_arguments.criterion
    testloader = processing_arguments.testloader
    device = processing_arguments.shared_arguments.device

    model.eval()
    correct = 0
    correct5 = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = torch.argmax(output, 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct5 += get_num_correct_pred(output, target, k=5)

    dataset_len = len(testloader.dataset)
    test_loss /= dataset_len
    acc = 100. * correct / len(testloader.dataset)
    acc5 = 100. * correct5 / len(testloader.dataset)

    print('\nTest set info: \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, dataset_len, acc))
    test_end = time.time()
    test_time = test_end - test_start

    return acc, acc5, test_time

import torch
import time

def final_evaluation(processing_arguments):

    model = processing_arguments.model
    criterion = processing_arguments.criterion
    testloader = processing_arguments.testloader
    device = processing_arguments.shared_arguments.device
    classes = processing_arguments.shared_arguments.classes

    model.eval()
    correct = 0
    correct5 = 0
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):

            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = torch.argmax(output, 1) #before: output.max(1, keepdim=True)[1]

            correct += get_num_correct_pred(output, target, k=1)
            correct5 += get_num_correct_pred(output, target, k=5)

            all_predictions.append(pred)
            all_targets.append(target)

    acc = 100. * correct / len(testloader.dataset)
    acc5 = 100. * correct5 / len(testloader.dataset)

    all_predictions_tensor = torch.FloatTensor().cpu()
    all_targets_tensor = torch.FloatTensor().cpu()
    for i in range(len(all_predictions)):
        single_pred = all_predictions[i].float().cpu()
        all_predictions_tensor = torch.cat([all_predictions_tensor,single_pred])
        single_target = all_targets[i].float().cpu()
        all_targets_tensor = torch.cat([all_targets_tensor,single_target])

    all_predictions_tensor.cpu()
    all_targets_tensor.cpu()

    cm_overall = confusion_matrix(all_predictions_tensor.view(-1), all_targets_tensor.view(-1), labels=classes)

    return acc, acc5, cm_overall
