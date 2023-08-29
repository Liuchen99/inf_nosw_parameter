import math
import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODEL_REGISTRY, ARCH_REGISTRY

from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class CLSIRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["img", "target"]

        self.network_names = ["netCLS"]
        self.networks = {}

        # pix: Pixel Loss; adv: Adversarial Loss; percep: Perceptual Loss
        self.loss_names = ["l_cls"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        self.niter = opt["train"]["niter"]

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))
        
        for name in defined_network_names:
            if nets_opt[name]["mode"] == "timm":
                setattr(self, name, self.build_network_timm(nets_opt[name]))
            else:
                setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)

        # define variable for training stage
        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.T_min, self.T_max = 1e-3, 1e1

            if train_opt.get("fmix"):
                self.fmix = ARCH_REGISTRY.get("FMix")()
    
    def Log_UP(self, K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / self.niter * epoch)]).float().cuda()
        
    def feed_data(self, data):
        self.img = data["img"].to(self.device)
        if self.opt["pre_upsample"] is not None:
            size = (self.opt["pre_upsample"], self.opt["pre_upsample"])
            self.img = F.interpolate(self.img, size, mode="bicubic", align_corners=True)
        if hasattr(self, "fmix"):
            self.img = self.fmix(self.img)

        self.target = data["target"].to(self.device)

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        t = self.Log_UP(self.T_min, self.T_max, step)
        if (t<1):
            k = 1 / t
        else:
            k = torch.tensor([1]).float().to(self.device)
        
        self.netCLS.module.set_kt(k, t)

        self.pred = self.netCLS(self.img)

        loss = self.losses.get("l_cls")(self.pred, self.target)
        loss_dict["l_cls"] = loss.item()

        self.set_optimizer(names=["netCLS"], operation="zero_grad")
        loss.backward()
        # self.clip_grad_norm(["netCLS"], self.max_grad_norm)
        self.set_optimizer(names=["netCLS"], operation="step")

        self.log_dict = loss_dict

    def test(self, test_data, crop_size=None):
        self.img = test_data["img"].to(self.device)
        if self.opt["pre_upsample"] is not None:
            size = (self.opt["pre_upsample"], self.opt["pre_upsample"])
            self.img = F.interpolate(self.img, size, mode="bicubic", align_corners=True)
        if test_data.get("target") is not None:
            self.test_target = test_data["target"].to(self.device)

        self.set_network_state(["netCLS"], "eval")
        with torch.no_grad():
            self.test_pred = self.netCLS(self.img)
        self.set_network_state(["netCLS"], "train")

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict["lr"] = self.src.detach()[0].float().cpu()
        # out_dict["sr"] = self.test_sr.detach()[0].float().cpu()
        return out_dict
    
    