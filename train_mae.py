"""
/************************************************************************
 MIT License
 Copyright (c) 2021 AI4CE Lab@NYU, MediaBrain Group@SJTU
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/
/**
 *  @file    train_codet.py
 *  @author  YIMING LI (https://roboticsyimingli.github.io/)
 *  @date    10/10/2021
 *  @version 1.0
 *
 *  @brief Training Pipeline of Collaborative BEV Detection
 *
 *  @section DESCRIPTION
 *
 *  This is official implementation for: NeurIPS 2021 Learning Distilled Collaboration Graph for Multi-Agent Perception
 *
 */
"""
import matplotlib
matplotlib.use('Agg')
import argparse

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.Dataset import V2XSIMDataset
from data.config import Config, ConfigGlobal
from utils.CoDetModule import *
from utils.loss import *
from utils.models import *

from yaml_parser import parse_config
import shutil

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
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config('train', binary=True, only_det=True)
    config_global = ConfigGlobal('train', binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch

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
    training_dataset = V2XSIMDataset(dataset_roots=[f'{args.data}/agent{i}' for i in range(5)], config=config,
                                     config_global=config_global, split='train')
    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    print("Training dataset size:", len(training_dataset))

    logger_root = args.logpath if args.logpath != '' else 'logs'

    if args.com == '':
        model = FaFNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'when2com':
        model = When2com(config, layer=args.layer, warp_flag=args.warp_flag)
    elif args.com == 'v2v':
        model = V2VNet(config, gnn_iter_times=args.gnn_iter_times, layer=args.layer, layer_channel=256)
    elif args.com == 'disco':
        model = DiscoNet(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'sum':
        model = SumFusion(config, layer=args.layer, kd_flag=args.kd_flag, layer1_channel=args.layer1_channel, layer2_channel=args.layer2_channel, layer3_channel=args.layer3_channel)
    elif args.com == 'mean':
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'max':
        model = MaxFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'cat':
        model = CatFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    elif args.com == 'agent':
        model = AgentWiseWeightedFusion(config, layer=args.layer, kd_flag=args.kd_flag)
    else:
        raise NotImplementedError('Invalid argument com:' + args.com)

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {'cls': SoftmaxFocalClassificationLoss(), 'loc': WeightedSmoothL1LocalizationLoss()}

    if args.kd_flag == 1:
        teacher = TeacherNet(config)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.to(device)
        faf_module = FaFModule(model, teacher, config, optimizer, criterion, args.kd_flag)
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher['epoch']
        faf_module.teacher.load_state_dict(checkpoint_teacher['model_state_dict'])
        print("Load teacher model from {}, at epoch {}".format(args.resume_teacher, start_epoch_teacher))
        faf_module.teacher.eval()
    else:
        faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
        
    

   
    if args.resume != None:
        model_save_path = args.resume[:args.resume.rfind('/')]

        log_file_name = os.path.join(model_save_path, 'log.txt')
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        faf_module.model.load_state_dict(checkpoint['model_state_dict'])
        faf_module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        faf_module.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    else:
        if need_log:
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, flag))

            shutil.copyfile(args.config_file, os.path.join(model_save_path, 'config.yaml'))
            shutil.copyfile('train_mae.py', os.path.join(model_save_path, 'train_mae.py'))
            shutil.copyfile('eval_mae.py', os.path.join(model_save_path, 'eval_mae.py'))
            shutil.copyfile(args.config_file, os.path.join(model_save_path, 'config.yaml'))
            shutil.copytree('data/', os.path.join(model_save_path, 'data'))
            shutil.copytree('utils/', os.path.join(model_save_path, 'utils'))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

        

    for epoch in range(start_epoch, num_epochs + 1):
        lr = faf_module.optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter('Total loss', ':.6f')
        running_loss_class = AverageMeter('classification Loss', ':.6f')  # for cell classification error
        running_loss_loc = AverageMeter('Localization Loss', ':.6f')  # for state estimation error

        faf_module.model.train()

        t = tqdm(training_data_loader)
        for sample in t:
            padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list, reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, \
            target_agent_id_list, num_agent_list, trans_matrices_list = zip(*sample)

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_agent = torch.stack(tuple(num_agent_list), 1)

            if flag == 'upperbound':
                padded_voxel_point = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            else:
                padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0)

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
            padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            data['bev_seq_teacher'] = padded_voxel_points_teacher.to(device)
            data['kd_weight'] = args.kd_weight

            loss = faf_module.step(data, batch_size)
            running_loss_disp.update(loss)
            # running_loss_class.update(cls_loss)
            # running_loss_loc.update(loc_loss)

            # if np.isnan(loss) or np.isnan(cls_loss) or np.isnan(loc_loss):
            #     print(f'Epoch {epoch}, loss is nan: {loss}, {cls_loss} {loc_loss}')
            #     sys.exit();

            t.set_description("Epoch {},     lr {}".format(epoch, lr))
            t.set_postfix(loss=running_loss_disp.avg)

        faf_module.scheduler.step()

        # save model
        if need_log:
            saver.write("{}\t{}\t{}\n".format(running_loss_disp, running_loss_class, running_loss_loc))
            saver.flush()
            if config.MGDA:
                save_dict = {'epoch': epoch,
                             'encoder_state_dict': faf_module.encoder.state_dict(),
                             'optimizer_encoder_state_dict': faf_module.optimizer_encoder.state_dict(),
                             'scheduler_encoder_state_dict': faf_module.scheduler_encoder.state_dict(),
                             'head_state_dict': faf_module.head.state_dict(),
                             'optimizer_head_state_dict': faf_module.optimizer_head.state_dict(),
                             'scheduler_head_state_dict': faf_module.scheduler_head.state_dict(),
                             'loss': running_loss_disp.avg}
            else:
                save_dict = {'epoch': epoch,
                             'model_state_dict': faf_module.model.state_dict(),
                             'optimizer_state_dict': faf_module.optimizer.state_dict(),
                             'scheduler_state_dict': faf_module.scheduler.state_dict(),
                             'loss': running_loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_log:
        saver.close()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml', type=str,
                        help='The path to the config file')
    args = parser.parse_args()
    args = parse_config(args.config_file, 'train')
    print(args)
    main(args)