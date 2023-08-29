import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
# from metrics import IQA
from models import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--opt", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "--root_path",
        help="experiment configure file name",
        default="./",
        type=str,
    )
    # distributed training
    parser.add_argument("--gpu", help="gpu id for multiprocessing training", type=str)
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()

    return args


def setup_dataloaer(opt, logger):

    if opt["dist"]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, opt["dist"])

            if opt["train"].get("niter") is not None:
                total_iters = opt["train"]["niter"]
                total_epochs = total_iters // (len(train_loader) - 1) + 1
            elif opt["train"].get("nepoch") is not None:
                total_epochs = opt["train"]["nepoch"]
                total_iters = total_epochs * len(train_loader)
            else:
                raise Exception("How many iter or epoch to train ???")    

            if rank == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), len(train_loader)
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )

        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt["dist"])
            if rank == 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))

    assert train_loader is not None
    assert val_loader is not None

    return train_set, train_loader, val_set, val_loader, total_iters, total_epochs


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    opt["dist"] = args.world_size > 1

    if opt["train"].get("resume_state", None) is None:
        print(opt["path"])
        util.mkdir_and_rename(
            opt["path"]["experiments_root"]
        )  # rename experiment folder if exists
        util.mkdirs(
            (path for key, path in opt["path"].items() if not key == "experiments_root")
        )
        # os.system("rm ./log")
        # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

    if opt["dist"]:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt, args))
    else:
        main_worker(0, 1, opt, args)


def main_worker(gpu, ngpus_per_node, opt, args):

    if opt["dist"]:
        if args.dist_url == "env://" and args.rank == -1:
            rank = int(os.environ["RANK"])

        rank = args.rank * ngpus_per_node + gpu
        print(
            f"Init process group: dist_url: \
            {args.dist_url}, world_size: {args.world_size}, rank: {rank}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )

        torch.cuda.set_device(gpu)

    else:
        rank = 0

    seed = opt["train"]["manual_seed"]
    if seed is None:
        util.set_random_seed(rank)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # setup tensorboard and val logger
    if rank == 0:
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger = SummaryWriter(log_dir="{}/tb_logger/".format(opt["path"]["experiments_root"]))

        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )

    # config loggers. Before it, the log will not work
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "train_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO if rank == 0 else logging.ERROR,
        screen=True,
        tofile=True,
    )

    logger = logging.getLogger("base")
    if rank == 0:
        logger.info(option.dict2str(opt))

    # create dataset
    (
        train_set,
        train_loader,
        val_set,
        val_loader,
        total_iters,
        total_epochs,
    ) = setup_dataloaer(opt, logger)

    # create model
    model = create_model(opt)

    # loading resume state if exists
    if opt["train"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = gpu
        resume_state = torch.load(
            opt["train"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )

        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers

    else:
        current_step = 0
        start_epoch = 0

    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    logger.info(
        "Total epoch: {:d}, iter: {:d}; batch_num: {:d}" .format(total_epochs, total_iters, len(train_loader))
    )

    data_time, iter_time = time.time(), time.time()
    avg_data_time = avg_iter_time = 0
    count = 0
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):

            current_step += 1
            count += 1
            if current_step > total_iters:
                break

            data_time = time.time() - data_time
            avg_data_time = (avg_data_time * (count - 1) + data_time) / count

            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if opt["train"]["train_mode"] == "iter-base":
                model.update_learning_rate(
                    current_step, warmup_iter=opt["train"]["warmup_iter"]
                )

            iter_time = time.time() - iter_time
            avg_iter_time = (avg_iter_time * (count - 1) + iter_time) / count

            # log
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = (
                    f"<epoch:{epoch:3d}, iter:{current_step:8,d}, "
                    f"lr:{model.get_current_learning_rate():.3e}> "
                )

                message += f'[time (data): {avg_iter_time:.3f} ({avg_data_time:.3f})] '
                for k, v in logs.items():
                    message += "{:s}: {:.4e}; ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank == 0:
                            tb_logger.add_scalar("train/%s" % (k), v, current_step)
                logger.info(message)

            # validation
            if current_step % opt["train"]["val_freq"] == 0:
                avg_results = validate(model, val_set, val_loader, epoch, current_step)

                # tensorboard logger
                if rank == 0:
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        for k, v in avg_results.items():
                            tb_logger.add_scalar("val/%s" % (k), v, current_step)

            # save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank == 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
            
            data_time = time.time()
            iter_time = time.time()

        if opt["train"]["train_mode"] == "epoch-base":
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

    if rank == 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of training.")
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger.close()


def validate(model, dataset, dist_loader, epoch, current_step):

    # calculate: loss, acc1, acc5
    acc1, acc5 = 0, 0

    for (idx, val_data) in enumerate(tqdm(dist_loader)):
        model.test(val_data)

        pred = model.test_pred
        target = model.test_target

        # acc1, acc5
        _, pred = pred.topk(5, 1, True, True)
        pred = pred.t().type_as(target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc1 += correct[:1].contiguous().view(-1).float().sum(0).item()
        acc5 += correct[:5].contiguous().view(-1).float().sum(0).item()

    acc1, acc5 = acc1 / len(dist_loader.dataset), acc5 / len(dist_loader.dataset)

    # log
    avg_results = {}
    avg_results["loss"] = 0
    avg_results["acc1"] = acc1
    avg_results["acc5"] = acc5
    
    message = " <epoch:{:3d}, iter:{:8,d}, Average sccores:\t".format(epoch, current_step)
    message += "loss: {:.6f};\tacc1: {:.4f};\tacc5: {:.4f}".format(0, acc1, acc5)
    logging.getLogger("val").info(message)

    torch.cuda.empty_cache()
    return avg_results


if __name__ == "__main__":
    main()
