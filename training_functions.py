import os
import numpy
import utils
import time
import torch
import numpy as np
import copy
from torch.autograd import Variable
from torch.nn import functional as F

class BalancedSoftmaxCE(torch.nn.Module):
    r"""
    References:
    Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
    Equation: Loss(x, c) = -log(\frac{n_c*exp(x)}{sum_i(n_i*exp(i)})
    """

    def __init__(self, num_class_list=None,device='cpu'):
        super(BalancedSoftmaxCE, self).__init__()
        self.bsce_weight = torch.FloatTensor(num_class_list).to(device)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = inputs + self.bsce_weight.unsqueeze(0).expand(inputs.shape[0], -1).log()
        loss = F.cross_entropy(input=logits, target=targets)
        return loss
class CDT(torch.nn.Module):
    r"""
    References:
    Class-Dependent Temperatures (CDT) Loss, Ye et al., Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning, arXiv 2020.
    Equation:  Loss(x, c) = - log(\frac{exp(x_c / a_c)}{sum_i(exp(x_i / a_i))}), and a_j = (N_max/n_j)^\gamma,
                where gamma is a hyper-parameter, N_max is the number of images in the largest class,
                and n_j is the number of image in class j.
    Args:
        gamma (float or double): to control the punishment to feature deviation.  For CIFAR-10, γ ∈ [0.0, 0.4]. For CIFAR-100
        and Tiny-ImageNet, γ ∈ [0.0, 0.2]. For iNaturalist, γ ∈ [0.0, 0.1]. We then select γ from several
        uniformly placed grid values in the range
    """
    def __init__(self, num_class_list=None,gamma=0.1,device='cpu'):
        super(CDT, self).__init__()
        self.gamma = gamma
        self.weight_list = torch.FloatTensor([(max(num_class_list) / i) ** self.gamma for i in num_class_list]).to(device)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = inputs / self.weight_list
        loss = F.cross_entropy(inputs, targets)
        return loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function
def train_model(model, dataloader,dataloadert,dataloaderv, criterion, optimizer, scheduler, num_epochs, params,class_num_list):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    trained = params['model_file']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    alpha = params['alpha']
    remix_kappa=params['remix_kappa']
    remix_tau = params['remix_tau']
    save_folder = str(numpy.char.replace(trained[:-3], 'nets', 'runs'))
    # Prep variables for weights and accuracy of the best model
    best_bacc = 0
    best_acc = 0
    best_acc_0 = 0
    best_acc_1 = 0
    best_acc_2 = 0

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # if epoch>100:
        #     class_weights = torch.tensor(
        #         numpy.array([min(class_num_list) / class_num_list[0], min(class_num_list) / class_num_list[1],
        #                      min(class_num_list) / class_num_list[2]]), dtype=torch.float).to(device)
        #     criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if alpha>0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels.long(), alpha, use_cuda=True)
                #print(lam)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                #print(outputs)
                if alpha>0:
                    if remix_kappa>0:
                        l_list = torch.empty(inputs.shape[0]).fill_(lam).float().to(device)
                        n_i=torch.tensor([class_num_list[target] for target in targets_a])
                        n_j = torch.tensor([class_num_list[target] for target in targets_b])
                        #n_i, n_j = class_num_list[targets_a], class_num_list[targets_b].float()
                        if lam < remix_tau:
                            l_list[n_i / n_j >= remix_kappa] = 0
                        if 1 - lam < remix_tau:
                            l_list[(n_i * remix_kappa) / n_j <= 1] = 1
                        loss = l_list * criterion(outputs, targets_a) + (1 - l_list) * criterion(outputs, targets_b)
                        loss = loss.mean()
                    else:
                        loss_func = mixup_criterion(targets_a, targets_b, lam)
                        loss = loss_func(criterion, outputs)
                else:
                    loss = criterion(outputs, labels.long())
                if epoch != num_epochs - 1:
                    loss.backward()
                    optimizer.step()
                # For keeping statistics
                running_loss += loss.item() * inputs.size(0)
                loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            # Some current stats
            loss_batch = loss.item()
            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'training:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Training/Loss', loss_accum, niter)
            batch_num = batch_num + 1

        epoch_loss = running_loss / dataset_size
        if board:
            writer.add_scalar('Training/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Training:\t Loss: {:.4f}'.format(epoch_loss))
        utils.print_both(txt_file, '')

        if epoch%1==0 or epoch==num_epochs-1:
            predst, labelst,probst,loss_train= calculate_predictions(model, dataloadert, params, criterion)
            acct = numpy.sum(predst == labelst) / len(labelst)
            acct_0 = numpy.sum(predst[labelst == 0] == 0) / len(predst[labelst== 0])
            acct_1 = numpy.sum(predst[labelst== 1] == 1) / len(predst[labelst == 1])
            #acct_2 = numpy.sum(predst[labelst== 2] == 2) / len(predst[labelst == 2])
            bacct=(acct_0+acct_1)/2
            #utils.print_both(txt_file, 'Training:\t ACC: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(bacct,acct,acct_0,acct_1,acct_2))
            utils.print_both(txt_file,
                             'Training:\t ACC: {:.4f} {:.4f} {:.4f} {:.4f}'.format(bacct, acct, acct_0, acct_1))
            if board:
                writer.add_scalar('Training/BACC' + '/Epoch', bacct, epoch + 1)
                writer.add_scalar('Training/ACC' + '/Epoch', acct, epoch + 1)
                writer.add_scalar('Training/ACC0' + '/Epoch', acct_0, epoch + 1)
                writer.add_scalar('Training/ACC1' + '/Epoch', acct_1, epoch + 1)
                #writer.add_scalar('Training/ACC2' + '/Epoch', acct_2, epoch + 1)

            preds,labels,probs,val_loss= calculate_predictions(model, dataloaderv, params,criterion)
            acc = numpy.sum(preds == labels) / len(labels)
            acc_0 = numpy.sum(preds[labels == 0] == 0) / len(preds[labels== 0])
            acc_1 = numpy.sum(preds[labels== 1] == 1) / len(preds[labels == 1])
            #acc_2 = numpy.sum(preds[labels== 2] == 2) / len(preds[labels == 2])
            bacc = (acc_0 + acc_1) / 2
            #utils.print_both(txt_file, 'Validation:\t ACC: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(bacc,acc,acc_0,acc_1,acc_2))
            utils.print_both(txt_file,
                             'Validation:\t ACC: {:.4f} {:.4f} {:.4f} {:.4f}'.format(bacc, acc, acc_0, acc_1))
            if bacc>best_bacc:
                preds_com = numpy.concatenate((preds, labels), 1)
                numpy.save(os.path.join(save_folder, 'preds_com.npy'), preds_com)
                best_bacc=bacc
                best_acc=acc
                best_acc_0 = acc_0
                best_acc_1 = acc_1
                #best_acc_2 = acc_2
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, trained)
            if board:
                writer.add_scalar('Validation/BACC' + '/Epoch', bacc, epoch + 1)
                writer.add_scalar('Validation/ACC' + '/Epoch', acc, epoch + 1)
                writer.add_scalar('Validation/ACC0' + '/Epoch', acc_0, epoch + 1)
                writer.add_scalar('Validation/ACC1' + '/Epoch', acc_1, epoch + 1)
                #writer.add_scalar('Validation/ACC2' + '/Epoch', acc_2, epoch + 1)
            # utils.print_both(txt_file,
            #     'Validation:\t Best_BACC: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(best_bacc, best_acc, best_acc_0, best_acc_1,best_acc_2))
            utils.print_both(txt_file,
                'Validation:\t Best_BACC: {:.4f} {:.4f} {:.4f} {:.4f}'.format(best_bacc, best_acc, best_acc_0, best_acc_1))
            utils.print_both(txt_file, 'Validation:\t Loss: {:.4f}'.format(val_loss))
            if board:
                writer.add_scalar('Validation/Loss' + '/Epoch', val_loss, epoch + 1)

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def calculate_predictions(model, dataloader, params,criterion):
    device=params['device']
    output_array = None
    probs_array = None
    label_array = None
    model.eval()
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(dim=1)
        probs = model(inputs)
        loss = criterion(probs, labels.long().squeeze(1))
        if params['remix_kappa']>0:
            loss = loss.mean()
        outputs = probs.data.max(1)[1].unsqueeze(dim=1)
        running_loss += loss.item() * inputs.size(0)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            probs_array = np.concatenate((probs_array, probs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)

        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()
            probs_array = probs.cpu().detach().numpy()
    dataset_size=output_array.shape[0]
    epoch_loss = running_loss / dataset_size

    # print(output_array.shape)
    return output_array, label_array,probs_array,epoch_loss

