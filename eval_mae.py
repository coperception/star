from unittest import result
from utils.models import *
from utils.loss import *
from utils.CoDetModule import *
from data.config import Config, ConfigGlobal
from data.Dataset import V2XSIMDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt 
import random

from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError

from yaml_parser import parse_config

matplotlib.use('Agg')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path



def main(args):
    config = Config('test', binary=True, only_det=True)
    config_global = ConfigGlobal('test', binary=True, only_det=True)


    num_workers = args.nworker
    epoch = 1
    batch_size = 1

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.bound == 'upperbound':
        flag = 'upperbound'
    elif args.bound == 'lowerbound':
        if args.com == 'when2com' and args.warp_flag:
            flag = 'when2com_warp'
        elif args.com in {'v2v', 'disco', 'sum', 'mean', 'max', 'cat', 'agent'}:
            flag = args.com
        else:
            flag = 'lowerbound'
    else:
        raise ValueError('not implement')

    config.flag = flag
    testing_dataset = V2XSIMDataset(dataset_roots=[f'{args.data}/agent{i}' for i in range(5)], config=config,
                                     config_global=config_global, split='test')
    testing_data_loader = DataLoader(
        testing_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    print("testing dataset size:", len(testing_dataset))


    if args.com == '':
        model = FaFNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'when2com':
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'v2v':
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times,
                       layer=args.layer, layer_channel=256)
    elif args.com == 'disco':
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'sum':
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'mean':
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'max':
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'cat':
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'agent':
        model = AgentWiseWeightedFusion(
            config, layer=args.layer, kd_flag=args.kd_flag)
    else:
        raise NotImplementedError('Invalid argument com:' + args.com)

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {'cls': SoftmaxFocalClassificationLoss(
    ), 'loc': WeightedSmoothL1LocalizationLoss()}

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(model, teacher, config,
                               optimizer, criterion, args.kd_flag)
        checkpoint_teacher = torch.load(args.resume_teacher)
        epoch_teacher = checkpoint_teacher['epoch']
        faf_module.teacher.load_state_dict(
            checkpoint_teacher['model_state_dict'])
        print("Load teacher model from {}, at epoch {}".format(
            args.resume_teacher, epoch_teacher))
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config,
                               optimizer, criterion, args.kd_flag)


    # model_save_path = args.resume[:args.resume.rfind('/')]

    # log_file_name = os.path.join(model_save_path, 'log.txt')

    checkpoint = torch.load(args.resume)
    epoch = checkpoint['epoch']
    faf_module.model.load_state_dict(checkpoint['model_state_dict'])
    faf_module.optimizer.load_state_dict(
        checkpoint['optimizer_state_dict'])
    faf_module.scheduler.load_state_dict(
        checkpoint['scheduler_state_dict'])
    
    resume_dir = os.path.dirname(args.resume)

    print("Load model from {}, at epoch {}".format(
        args.resume, epoch))

    print("Epoch {} Evaluation".format(epoch))

    faf_module.model.eval()
    
    rmse = RootMeanSquaredError()
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    
    eval_dir = os.path.join(resume_dir, "eval", "Epoch_%s" % epoch)
    check_folder(eval_dir)
    
    sample_dir =  os.path.join(eval_dir, "samples")
    check_folder(sample_dir)
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(testing_data_loader)):
            
            padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, \
                target_agent_id_list, num_agent_list, trans_matrices_list = zip(
                    *sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_agent = torch.stack(tuple(num_agent_list), 1)

            if flag == 'upperbound':
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0)
            else:
                padded_voxel_point = torch.cat(
                    tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {'bev_seq': padded_voxel_point.to(device),
                    'labels': label_one_hot.to(device),
                    'reg_targets': reg_target.to(device),
                    'anchors': anchors_map.to(device),
                    'reg_loss_mask': reg_loss_mask.to(device).type(dtype=torch.bool),
                    'vis_maps': vis_maps.to(device),
                    'target_agent_ids': target_agent_id.to(device),
                    'num_agent': num_agent.to(device),
                    'trans_matrices': trans_matrices}

            # if args.kd_flag == 1:
            padded_voxel_points_teacher = torch.cat(
                tuple(padded_voxel_points_teacher_list), 0)
            data['bev_seq_teacher'] = padded_voxel_points_teacher.to(device)
            data['kd_weight'] = args.kd_weight
            
            

            result = faf_module.model(data['bev_seq'], data['trans_matrices'], data['num_agent'], batch_size=batch_size)
            
            labels = data['bev_seq_teacher'].permute(0, 1, 4, 2, 3).squeeze()
            # print(input.min(), input.max())
            # print((labels - result).sum())
            rmse.update((result, labels))
            mse.update((result, labels))
            mae.update((result, labels))
            
            # loss = faf_module.step(data, batch_size=batch_size)

                        
            # for num in range(args.sample_num):
            target_dir = os.path.join(sample_dir, str(idx))
            check_folder(target_dir)
            
            if args.visualization:
                for i in range(5):
                    bev_single_view = data['bev_seq'].permute(0, 1, 4, 2, 3).squeeze()[i]
                    bev_single_view, index = torch.max(bev_single_view, dim=0)
                    bev_single_view = bev_single_view.cpu().numpy()
                        
                    bev_output = result[i]
                    bev_output, index = torch.max(bev_output, dim=0)
                    bev_output = bev_output.cpu().numpy()
                
                    bev_multi_view = torch.cat(tuple(padded_voxel_points_teacher_list), 0).permute(
                    0, 1, 4, 2, 3).squeeze()[i]
                    bev_multi_view, index = torch.max(bev_multi_view, dim=0)
                    bev_multi_view = bev_multi_view.cpu().numpy()
                    
                    # output_thresh = 0.9 if bev_output.min() > 0 else 0
                    bev_binary_output = bev_output > args.output_thresh 
                    plt.imsave(os.path.join(target_dir, 'agent_%s_input.png' % i), bev_single_view, vmin=0, vmax=1)
                    plt.imsave(os.path.join(target_dir, 'agent_%s_binary_output.png' % i), bev_binary_output, vmin=0, vmax=1)
                    plt.imsave(os.path.join(target_dir, 'agent_%s_output.png' % i), bev_output, vmin=0, vmax=np.max(bev_output))
                    plt.imsave(os.path.join(target_dir, 'agent_%s_multiview_gt.png' % i), bev_multi_view, vmin=0, vmax=1)
            
            if idx == args.sample_num - 1 & args.sample_num > 0:
                break
        print("RMSE: ", rmse.compute())
        print("MSE: ", mse.compute())
        print("MAE: ", mae.compute())
        
        if args.log:
            log_file_name = os.path.join(eval_dir, 'eval_log.txt')
            saver = open(log_file_name, "a")
            saver.write("output_thresh: {}\n".format(args.output_thresh))
            saver.write("RMSE: {}\n".format(rmse.compute()))
            saver.write("MSE: {}\n".format(mse.compute()))
            saver.write("MAE: {}".format(mae.compute()))
        
                    
          


if __name__ == "__main__":
    setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='logs/sum_bceloss_sigmoidFixed_fulayer3_32x32_32channel_steplr_1_0.8/sum/epoch_20.pth', type=str,
                        help='The path to the saved model that is loaded to resume training')
    args = parser.parse_args()
    resume_path = os.path.dirname(args.resume)
    print(resume_path)
    args = parse_config(os.path.join(resume_path, 'config.yaml'), 'eval', args.resume)
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    print(args)
    main(args)
