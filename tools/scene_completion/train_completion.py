import argparse
import os

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets.V2XSimDet import V2XSimDet, MultiTempV2XSimDet

from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.models.det import *
from coperception.models.transformers import multiagent_mae
from coperception.utils import AverageMeter

# from mae script -----------
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import coperception.utils.maeutil.misc as misc
# from maeutil.misc import NativeScalerWithGradNormCount as NativeScaler
# import models_mae
# import temporal_mae
# import multiagent_mae
# import wandb
# from einops import rearrange
import coperception.utils.maeutil.lr_sched as lr_sched
# import matplotlib.pyplot as plt
# ----------------------------

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker
    start_epoch = 1
    batch_size = args.batch
    num_agent = args.num_agent

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    if args.com in {"mean", "max", "cat", "sum", "v2v", "ind_mae", "joint_mae"}:
        flag = args.com
    else:
        raise ValueError(f"com: {args.com} is not supported")

    config.flag = flag

    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)

    # training_dataset = V2XSimDet(
    #     dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
    #     config=config,
    #     config_global=config_global,
    #     split="train",
    #     bound="both",
    #     kd_flag=args.kd_flag,
    #     no_cross_road=args.no_cross_road,
    # )
    training_dataset = MultiTempV2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="train",
        bound="both",
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
        time_stamp = args.time_stamp
    )
    training_data_loader = DataLoader(
        training_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    print("Training dataset size:", len(training_dataset))

    logger_root = args.logpath if args.logpath != "" else "logs"

    if args.no_cross_road:
        num_agent -= 1

    if args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            train_completion=True
        )
    elif args.com == "joint_mae" or  args.com == "ind_mae":
        # Juexiao added for mae
        model = multiagent_mae.__dict__[args.mae_model](norm_pix_loss=args.norm_pix_loss, time_stamp=args.time_stamp, mask_method=args.mask)
        # also include individual reconstruction: reconstruct then aggregate
    else:
        raise NotImplementedError("Invalid argument com:" + args.com)

    model = nn.DataParallel(model)
    model = model.to(device)
    # Juexiao added for mae
    if args.com == "ind_mae" or args.com == "joint_mae":
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    faf_module = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = check_folder(logger_root)
    model_save_path = check_folder(os.path.join(model_save_path, flag))

    if args.no_cross_road:
        model_save_path = check_folder(os.path.join(model_save_path, "no_cross"))
    else:
        model_save_path = check_folder(os.path.join(model_save_path, "with_cross"))

    if args.resume:
        model_save_path = args.resume[: args.resume.rfind("/")]

        log_file_name = os.path.join(model_save_path, "log_completion.txt")
        saver = open(log_file_name, "a")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        faf_module.model.load_state_dict(checkpoint["model_state_dict"])
        faf_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        faf_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
    else:
        if need_log:
            log_file_name = os.path.join(model_save_path, "log_completion.txt")
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    for epoch in range(start_epoch, num_epochs + 1):
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter("Total loss", ":.6f")
        running_loss_class = AverageMeter(
            "classification Loss", ":.6f"
        )  # for cell classification error
        running_loss_loc = AverageMeter(
            "Localization Loss", ":.6f"
        )  # for state estimation error

        faf_module.model.train()

        t = tqdm(training_data_loader)
        for data_iter_step, sample in enumerate(t):
            (
                padded_voxel_point_list,
                padded_voxel_point_next_list, # time_stamp=1: tuple(num_agent x [batchsize, 0]), >1: tuple(num_agent x [batchsize, time_stamp-1, H,W,C])
                padded_voxel_points_teacher_list,
                label_one_hot_list,
                reg_target_list,
                reg_loss_mask_list,
                anchors_map_list,
                vis_maps_list,
                target_agent_id_list,
                num_agent_list,
                trans_matrices_list,
            ) = zip(*sample)
            # accomodate multitemp dataset
            # timestamp 1, 2, 3 and etc

            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_id = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1) # num_agent

            if args.no_cross_road:
                num_all_agents -= 1

            padded_voxel_point = torch.cat(tuple(padded_voxel_point_list), 0) #[num_agent x batch_size, H, W, C]

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_point.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "vis_maps": vis_maps.to(device),
                "target_agent_ids": target_agent_id.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices,
            }
            padded_voxel_points_teacher = torch.cat(
                tuple(padded_voxel_points_teacher_list), 0
            )
            data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)

            if args.com == "ind_mae" or args.com == "joint_mae":
                # multi timestamp input bev supported in the data
                data["bev_seq_next"] = torch.cat(tuple(padded_voxel_point_next_list), 0).to(device)
                # adjust learning rate for mae
                lr_sched.adjust_learning_rate(faf_module.optimizer, data_iter_step / len(training_data_loader) + epoch, args)
                loss, _ = faf_module.step_mae_completion(data, batch_size, args.mask_ratio, trainable=True)
            else:
                loss, _ = faf_module.step_completion(data, batch_size, trainable=True)
            running_loss_disp.update(loss)
            curr_lr = faf_module.optimizer.param_groups[0]['lr']
            t.set_description("Epoch {},     lr {}".format(epoch, curr_lr))
            t.set_postfix(loss=running_loss_disp.avg)

        if (args.com != "ind_mae" and args.com != "joint_mae"):
            faf_module.scheduler.step() ## avoid this affects mae training

        # save model
        if need_log:
            saver.write(
                "{}\t{}\t{}\n".format(
                    running_loss_disp, running_loss_class, running_loss_loc
                )
            )
            saver.flush()
            if config.MGDA:
                save_dict = {
                    "epoch": epoch,
                    "encoder_state_dict": faf_module.encoder.state_dict(),
                    "optimizer_encoder_state_dict": faf_module.optimizer_encoder.state_dict(),
                    "scheduler_encoder_state_dict": faf_module.scheduler_encoder.state_dict(),
                    "head_state_dict": faf_module.head.state_dict(),
                    "optimizer_head_state_dict": faf_module.optimizer_head.state_dict(),
                    "scheduler_head_state_dict": faf_module.scheduler_head.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": faf_module.model.state_dict(),
                    "optimizer_state_dict": faf_module.optimizer.state_dict(),
                    "scheduler_state_dict": faf_module.scheduler.state_dict(),
                    "loss": running_loss_disp.avg,
                }
            torch.save(
                save_dict,
                os.path.join(
                    model_save_path, "completion_epoch_" + str(epoch) + ".pth"
                ),
            )

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--com", default="", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )

    ## ----- mae args added by Juexiao -----
    parser.add_argument(
        "--mask_ratio", default=0.50, type=float, help="The input mask ratio"
    )
    parser.add_argument(
        "--mask", default="random", type=str, choices=["random", "complement"], help="Used in MAE training"
    )
    parser.add_argument(
        "--warmup_epochs", default=4, type=int, help="The number of warm up epochs"
    )
    parser.add_argument(
        "--time_stamp", default=2, type=int, help="The total number of time stamp to use"
    )
    parser.add_argument(
        "--mae_model", default="fusion_bev_mae_vit_base_patch8", type=str, 
        help="The mae model to use"
    )
    parser.add_argument(
        "--norm_pix_loss", default=False, type=bool, help="Whether normalize target pixel value for loss"
    )
    parser.add_argument(
        "--weight_decay", default=0.05, type=float, help="Used in MAE training"
    )
    ## ----------------------

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)