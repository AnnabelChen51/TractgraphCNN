import time
import numpy
import whitematteranalysis as wma
import abcd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import fnmatch
import training_functions
import nets
import utils
from torch.utils.tensorboard import SummaryWriter
import pandas
import h5py
import sys
import random
from sklearn.model_selection import train_test_split
from training_functions import BalancedSoftmaxCE,CDT

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-inFile', action="store", dest="inputDirectory", default='data', help='A file of features.')
    # parser.add_argument(
    #     '-indirv', action="store", dest="inputDirectoryv",
    #     default="../dataFolder/HCPTestingData/tractography_yc/validation",
    #     help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    # parser.add_argument(
    #     '-indirt', action="store", dest="inputDirectoryt", default="../dataFolder/HCPTestingData/tractography_yc/test2",
    #     help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outdir', action="store", dest="outputDirectory", default="try",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-dis_file', action="store", dest="DisFile", default="data/dis_sort_roi2dis2000.npy",
        help='folder of edge definitions.')
    parser.add_argument('--task', default='sex', choices=['external','internal','sex'], help='task')
    parser.add_argument('--type', default='binary', choices=['multi-class', 'binary'], help='task')
    parser.add_argument('--feature', default=['Nos','FA1'], nargs='+',help='task')
    parser.add_argument('--CUDA_id', default='0', choices=['0', '1','2','3'], help='choose cuda')
    parser.add_argument('--data_id', default='0', choices=['0','1', '2', '3'], help='data_id')
    parser.add_argument('--dataset', default='Classification', choices=['Classification','Classification_sample'], help='data resample')
    parser.add_argument('--norm', default=True, type=str2bool, help='whether to do feature normalization')
    parser.add_argument('--channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--epochs', default=400, type=int, help='training epochs')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--net_architecture', default='TractGraphormer', choices=['CNN_1D','DGCNN','TractGraphormer','TractGraphormerG','GCN','GCN1','PointNet','PointTrans','PointTrans1','Braingnn','DGCNNG'], help='network architecture used')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight', default=0.000, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--printing_frequency', default=1, type=int, help='training stats printing frequency')
    parser.add_argument('--seed', default=0, type=int, help='seed for random')
    parser.add_argument('--alpha', default=0, type=float, help='interpolation strength (uniform=1., ERM=0. for mmix up)')
    parser.add_argument('--remix_kappa', default=0, type=float,help='parameter for redmix')
    parser.add_argument('--remix_tau', default=0, type=float, help='parameter for redmix')
    parser.add_argument('--loss', default='CE', choices=['CE','CS_CE','SCS_CE','BSCE','CDT'], help='loss type')
    parser.add_argument('--sigma', default=0, type=float, help='parameter for Guassian noise')
    parser.add_argument('--k', default=20, type=int, help='k for dgcnn')
    args = parser.parse_args()

    #setup_seed(args.seed)
    torch.manual_seed(args.seed)
    board = args.tensorboard
    dataset = args.dataset
    batch = args.batch_size
    rate = args.rate
    weight = args.weight
    sched_step = args.sched_step
    epochs = args.epochs
    print_freq = args.printing_frequency
    sched_gamma = args.sched_gamma
    channels=args.channels
    features=args.feature

    # Directories
    # Create directories structure
    sub_folder=args.outputDirectory
    txt_folder=os.path.join('reports',sub_folder)
    model_folder = os.path.join('nets', sub_folder)
    event_folder=os.path.join('runs',sub_folder)
    dirs = [event_folder, txt_folder, model_folder]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture
    model_name = args.net_architecture
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    reports_list = sorted(os.listdir(txt_folder), reverse=True)
    if reports_list:
        for file in reports_list:
            # print(file)
            if fnmatch.fnmatch(file, model_name + '_0'+'*'):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1
    isDebug = True if sys.gettrace() else False
    if isDebug==True:
        idx=0
    print('save_id', idx)
    # Base filename
    name = model_name + '_' + str(idx).zfill(3)
    name_net = name+'.pt'
    name_txt = name + '.txt'
    name_txt = os.path.join(txt_folder, name_txt)
    name_net = os.path.join(model_folder, name_net)
    workers = 0

    f = open(name_txt, 'w')
    params = {'model_file': name_net}
    params['txt_file'] = f

    # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
    try:
        os.system("rm -rf "+event_folder + name)
    except:
        pass
    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter(event_folder +'/'+ name)
        params['writer'] = writer
    else:
        params['writer'] = None
    params['batch'] = batch
    params['print_freq'] = print_freq
    params['Learning rate']=rate
    params['alpha'] = args.alpha
    params['remix_kappa'] = args.remix_kappa
    params['remix_tau'] = args.remix_tau

    utils.print_both(f,str(args))
    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    utils.print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    utils.print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    utils.print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    utils.print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of input channels:\t" + str(channels)
    utils.print_both(f, tmp)

    data_dir = args.inputDirectory
    #data_dirv = args.inputDirectoryv
    #data_dirt=args.inputDirectoryt
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    utils.print_both(f, tmp)
    if args.data_id=='0':
        x_arrays=numpy.load(os.path.join(data_dir,'feature_array_dis.npy'))  #9344*1516*5
        Nos = x_arrays[:,:,0]
        Nos_sum=numpy.sum(Nos,1,keepdims=True)
        Nos_norm=Nos/Nos_sum
        x_arrays[:,:,0]=Nos_norm
        gt = numpy.load(os.path.join(data_dir, 'gt_array6.npy'))  #9344*5
    else:
        x_arrays=numpy.load(os.path.join(data_dir,'feature_array{}.npy'.format(str(args.data_id))))  #9344*1516*5
        gt = numpy.load(os.path.join(data_dir, 'gt_array{}.npy'.format(str(args.data_id))))
    indexs_empty=numpy.where(numpy.sum(gt,1)==0)
    x_arrays=numpy.delete(x_arrays,indexs_empty,axis=0)
    gt = numpy.delete(gt, indexs_empty, axis=0)
    if args.task=='external':
        y = gt[:,1]
    elif args.task == 'internal':
        y = gt[:, 3]
    elif args.task == 'sex':
        y = gt[:, 5]
    if len(args.feature)==1:
        if args.feature==['all']:
            X=x_arrays
            channels=5
        else:
            if args.feature==['Nos']:
                X = x_arrays[:,:,0]
            elif args.feature==['FA1']:
                X = x_arrays[:,:, 1]
            elif args.feature == ['MD1']:
                X = x_arrays[:,:, 2]
            elif args.feature==['FA2']:
                X = x_arrays[:, :,3]
            elif args.feature == ['MD2']:
                X = x_arrays[:, :,4]
            X=numpy.expand_dims(X,axis=2)
            channels=1
    elif len(args.feature)==2:
        X = x_arrays[:,:,0:2]
        channels = 2

    if args.net_architecture=='PointNet' or args.net_architecture == 'PointTrans'\
            or args.net_architecture == 'PointTrans1':
        mean_coor=numpy.load('data/mean_coor.npy')
        mean_coors=numpy.repeat(numpy.expand_dims(mean_coor,axis=0),X.shape[0],axis=0)
        X = numpy.concatenate((mean_coors, X), axis=2)
        channels = 5

    if args.task=='external' and args.type=='binary':
        # inds=numpy.where((y==0) | (y==2))[0]
        # X=X[inds]
        # y=y[inds]
        y[y==2]=1
    # X=X[:80,:,:]
    # y = y[:80]

    print(args.seed)
    print(X.shape)
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    x_tv, x_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=1/8, random_state=args.seed)

    if args.norm:
        #normlize
        #feat_max=numpy.max(numpy.max(x_train, 0),0)
        # feat_min = numpy.min(numpy.min(x_train, 0), 0)
        # feat_md=feat_max-feat_min
        # x_train = ((x_train - feat_min) / feat_md)
        # x_test = ((x_test - feat_min) / feat_md)
        x_train_flat=numpy.reshape(X,(-1,x_train.shape[2]))
        feat_mean = numpy.mean(x_train_flat, 0)
        feat_std = numpy.std(x_train_flat, 0)
        # feat_mean= numpy.mean(numpy.mean(x_train, 0), 0)
        # feat_std = numpy.std(numpy.std(x_train, 0), 0)
        x_train = ((x_train - feat_mean) / feat_std)
        x_test = ((x_test - feat_mean) / feat_std)

    if args.type=='binary':
        y_train_0=len(numpy.where(y_train==0)[0])
        y_train_1 = len(numpy.where(y_train == 1)[0])
        class_num_list=[y_train_0,y_train_1]
        num_class=2
    else:
        y_train_0=len(numpy.where(y_train==0)[0])
        y_train_1 = len(numpy.where(y_train == 1)[0])
        y_train_2 = len(numpy.where(y_train == 2)[0])
        class_num_list=[y_train_0,y_train_1,y_train_2]
        num_class = 3

    std_cluster=numpy.std(x_train,0)
    sigmas=args.sigma*std_cluster
    if args.dataset=='Classification':
        dataset = abcd.Classification(x_train, y_train,sigma=sigmas)
    elif args.dataset=='Classification_sample':
        #dataset = abcd.Classification_sample(x_train, y_train,3)
        dataset = abcd.Classification_sample1(x_train, y_train, min(class_num_list))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers,drop_last=True)
    dataloadert = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,drop_last=False)

    dataset_size = len(dataset)
    tmp = "Training set size:\t" + str(dataset_size)
    utils.print_both(f, tmp)

    assert len(x_test)==len(y_test)
    datasetv = abcd.Classification(x_test, y_test)
    dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=batch, shuffle=False, num_workers=workers,drop_last=False)
    dataset_sizev = len(datasetv)
    tmp = "Validation set size:\t" + str(dataset_sizev)
    utils.print_both(f, tmp)

    params['dataset_size'] = dataset_size
    # GPU check
    #device=torch.device("cuda")
    device = torch.device("cuda:{}".format(args.CUDA_id) if torch.cuda.is_available() else "cpu")
    tmp = "\nPerforming calculations on:\t" + str(device)
    utils.print_both(f, tmp + '\n')
    params['device'] = device

    # Evaluate the proper model
    if args.net_architecture=='CNN_1D':
        to_eval = "nets." + model_name + "(input_channel=channels,input_len=X.shape[1],num_classses=num_class)"
    elif args.net_architecture=='PointNet':
        to_eval = "nets." + model_name + "(input_channel=channels,num_classses=num_class)"
    elif args.net_architecture=='DGCNN' or args.net_architecture=='DGCNNs' or args.net_architecture=='TractGraphormer':
        idx_matrix = numpy.load('data/distance_id_sort_dis.npy')
        idx = idx_matrix[:, :args.k]
        idx=torch.from_numpy(idx).to(device)
        idx = idx.repeat(batch, 1, 1)
        to_eval = "nets." + model_name + "(input_channel=channels,input_len=X.shape[1],num_classses=num_class,k=args.k,idx=idx)"
    elif args.net_architecture == 'DGCNNAG' or args.net_architecture == 'DGCNNG' or args.net_architecture == 'DGCNNAMG' or args.net_architecture=='TractGraphormerG':
        idx = numpy.load(args.DisFile)
        print('load ' + args.DisFile)
        # idx1 =numpy.load('data/dis_sort_roi2.npy') #roi
        # nums=[]
        # for i in range(953):
        #     a=idx[i,:]
        #     num=len(numpy.unique(a))
        #     nums.append(num)
        # nums=numpy.array(nums)
        # weights=numpy.load('data/weight_sort_tract.npy')
        k = idx.shape[1]
        idx = torch.from_numpy(idx).long().to(device)
        idx = idx.repeat(batch, 1, 1)
        # weights = torch.from_numpy(weights).float().to(device)
        if args.net_architecture == 'DGCNNG':
            model_name = 'DGCNN'
        elif args.net_architecture == 'DGCNNAG':
            model_name = 'DGCNNA'
        elif args.net_architecture == 'DGCNNAMG':
            model_name = 'DGCNNAM'
        to_eval = "nets." + model_name + "(input_channel=channels,input_len=X.shape[1],num_classses=num_class,k=k,idx=idx)"
    elif args.net_architecture == 'GCN':
        edge_index=numpy.load('data/edge_indexes_40.npy')
        edge_weight = numpy.load('data/edge_weights_40.npy')
        edge_index=torch.from_numpy(edge_index.T).long().to(device)
        edge_weight = torch.from_numpy(edge_weight).float().to(device)
    elif args.net_architecture == 'Braingnn':
        edge_index=numpy.load('data/edge_indexes_40.npy')
        edge_weight = numpy.load('data/edge_weights_40.npy')
        edge_index=torch.from_numpy(edge_index.T).long().to(device)
        edge_weight = torch.from_numpy(numpy.expand_dims(edge_weight,1)).float().to(device)
        batchg=numpy.repeat(numpy.expand_dims(numpy.array(range(batch)),0),X.shape[1],axis=0).T.flatten()
        pos=numpy.tile(numpy.eye(X.shape[1]),(batch,1))
        batchg = torch.from_numpy(batchg).long().to(device)
        pos = torch.from_numpy(pos).float().to(device)
        to_eval = "nets." + model_name + "(channels,X.shape[1],num_class,edge_index, edge_weight, batchg, pos)"
    elif args.net_architecture == 'GCN1':
        As=[]
        for thr in [30,40,50,60]:
            adj_mat=numpy.load('data/adj_mat_{}.npy'.format(thr))
            adj_mat = adj_mat + numpy.eye(adj_mat.shape[0])
            # adj_mat = numpy.matrix(adj_mat+numpy.eye(adj_mat.shape[0]))
            # dgreee_mat=numpy.array(numpy.sum(adj_mat, axis=0))[0]
            # degree_mat = numpy.matrix(numpy.diag(dgreee_mat))
            # adj_mat=numpy.array(degree_mat**-1*adj_mat)
            Nn=adj_mat.shape[0]
            A=numpy.zeros((Nn*batch,Nn*batch),dtype='float32')
            for i in range(batch):
                A[i*Nn:(i+1)*Nn,i*Nn:(i+1)*Nn]=adj_mat
            A = torch.from_numpy(A).to(device)
            As.append(A)
        to_eval = "nets." + model_name + "(input_channel=channels,num_classes=num_class,A=As)"
    elif args.net_architecture == 'PointTrans' or args.net_architecture == 'PointTrans1':
        to_eval = "nets." + model_name + "(args,output_channels=num_class)"

    model = eval(to_eval)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total', total_num, 'Trainable', trainable_num)
    get_parameter_number(model)

    model = model.to(device)

    #load pretrained model
    model_dict = model.state_dict()
    model_pre=torch.load('/home/yuqian/hdrive/tabular-dl-revisiting-models/reproduce/ft_transformer_EP/checkpoint.pt')
    pretrained_dict=model_pre['model']
    dict_dis=['transformer.layers.0.key_compression.weight','transformer.layers.0.value_compression.weight','transformer.head.weight','transformer.head.bias']
    loaded_dict = {('transformer.'+ k): v for k, v in pretrained_dict.items() if ('transformer.'+ k) in model_dict and ('transformer.'+ k) not in dict_dis}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)

    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
    #criteria = nn.NLLLoss(size_average=True)
    if args.loss=='CE':
        if args.remix_kappa>0:
            criteria = nn.CrossEntropyLoss(reduction='none')
        else:
            criteria = nn.CrossEntropyLoss()
    elif args.loss=='CS_CE':
        if args.type=='binary':
            class_weights = torch.tensor(numpy.array([min(class_num_list) / class_num_list[0],
                                                      min(class_num_list) / class_num_list[1]]), dtype=torch.float).to(device)
        else:
            class_weights = torch.tensor(numpy.array([min(class_num_list) / class_num_list[0], min(class_num_list) / class_num_list[1],
                                                      min(class_num_list) / class_num_list[2]]), dtype=torch.float).to(device)
        criteria = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'SCS_CE':
        class_weights = torch.tensor(numpy.array([(min(class_num_list) / class_num_list[0]) ** 0.5, (min(class_num_list) / class_num_list[1])** 0.5,
                        (min(class_num_list) / class_num_list[2]) ** 0.5]), dtype=torch.float).to(device)
        criteria = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss=='BSCE':
        criteria = BalancedSoftmaxCE(num_class_list=[class_num_list[0],class_num_list[1],class_num_list[2]],device=device)
    elif args.loss == 'CDT':
        criteria = CDT(num_class_list=[class_num_list[0], class_num_list[1], class_num_list[2]], gamma = 0.4, device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=args.weight)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    training_functions.train_model(model,dataloader,dataloadert,dataloaderv,criteria,optimizer,scheduler,epochs,params,class_num_list)
