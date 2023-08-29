import argparse
import logging
import os.path
import sys
import time
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from scipy.io import loadmat
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import utils as util
import utils.option as option
from data import create_dataloader, create_dataset

from models import create_model
# from sim.aDobe_s9_python_sim_class import AdobeS9CimSim

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


def main():
    args = parse_args()
    opt = option.parse(args.opt, args.root_path, is_train=False)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    opt["dist"] = args.world_size > 1
    print(opt["path"])
    util.mkdirs(
        (path for key, path in opt["path"].items() if not key == "experiments_root")
    )

    # os.system("rm ./result")
    # os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

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
            f"Init process group: dist_url: {args.dist_url}, world_size: {args.world_size}, rank: {rank}"
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

    torch.backends.cudnn.benchmark = True

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"] + "_rank{}".format(rank),
        level=logging.INFO,
        screen=True,
        tofile=True,
    )

    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # Create test dataset and dataloader
    test_datasets = []
    test_loaders = []

    for phase, dataset_opt in sorted(opt["datasets"].items()):

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt["dist"])

        if rank == 0:
            logger.info(
                "Number of test images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(test_set)
                )
            )
        test_datasets.append(test_set)
        test_loaders.append(test_loader)

    # load pretrained model by default
    model = create_model(opt)
    # handle = AdobeS9CimSim()
    # handle.pth_model_handle(model)

    for test_dataset, test_loader in zip(test_datasets, test_loaders):

        test_set_name = test_dataset.opt["name"]
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)

        if rank == 0:
            logger.info("\nTesting [{:s}]...".format(test_set_name))
            util.mkdir(dataset_dir)

        validate(
            model,
            test_dataset,
            test_loader,
            test_set_name,
            logger,
        )

def validate(model, dataset, dist_loader, test_set_name, logger):

    # calculate: loss, acc1, acc5
    acc1, acc5 = 0, 0

    for (idx, val_data) in enumerate(dist_loader):
        model.test(val_data)

        pred = model.test_pred
        target = model.test_target
        # acc1, acc5
        _, pred = pred.topk(5, 1, True, True)
        pred = pred.t().type_as(target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc1_item = correct[:1].contiguous().view(-1).float().sum(0).item()
        acc5_item = correct[:5].contiguous().view(-1).float().sum(0).item()
        acc1 += acc1_item
        acc5 += acc5_item

        print("acc1: {:.4f} \tacct: {:.4f}".format(acc1_item / val_data["img"].size(0), acc5_item / val_data["img"].size(0)))

    acc1, acc5 = acc1 / len(dist_loader.dataset), acc5 / len(dist_loader.dataset)

    # log
    avg_results = {}
    avg_results["loss"] = 0
    avg_results["acc1"] = acc1
    avg_results["acc5"] = acc5

    
    message = "Average sccores for {}\n".format(test_set_name)
    message += "loss: {:.6f};\tacc1: {:.4f};\tacc5: {:.4f}".format(0, acc1, acc5)
    logger.info(message)


if __name__ == "__main__":
    main()
