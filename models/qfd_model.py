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
class QFDModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["img", "target"]

        self.network_names = ["netCLS", "netTeach"]
        self.networks = {}

        # pix: Pixel Loss; adv: Adversarial Loss; percep: Perceptual Loss
        self.loss_names = ["l_cls", "l_teach"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

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

            # fix the params for netTeach
            self.set_requires_grad(["netTeach"], False)
            
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            if train_opt.get("fmix"):
                self.fmix = ARCH_REGISTRY.get("FMix")()

            self.set_network_state(["netTeach"], "eval")
        
    def feed_data(self, data):
        self.img = data["img"].to(self.device)
        if self.opt["pre_upsample"] is not None:
            size = (self.opt["pre_upsample"], self.opt["pre_upsample"])
            self.img_teach = F.interpolate(self.img, size, mode="bicubic", align_corners=True)
        else:
            self.img_teach = self.img
        if hasattr(self, "fmix"):
            self.img = self.fmix(self.img)

        self.target = data["target"].to(self.device)
    
    def quant(self, x, bit=3, low=0, high=6.1320):
        # Tmax = torch.max(x).detach()
        # Tmin = torch.min(x).detach()
        # T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        # T = torch.clamp(T, 1e-10, 255.)
        # x = torch.clamp(x, -T, T) / T       # TODO: this may cause errors ?!
        # n = float(2 ** bit -1)
        # activations_q = torch.round(x * n) / n
        # return activations_q
        # print(x.max(), x.min())

        n = float(2 ** bit -1)
        x = torch.clamp((x-low)/(high-low), 0, 1)   # [min, max]->[0, 1]
        x = torch.round(x * n) / n                  # [0, 1]->[0, n]->[0, 1]
        # Tmax = torch.max(x).detach()
        # Tmin = torch.min(x).detach()
        # T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        # T = torch.clamp(T, 1e-10, 255.)
        # x = torch.clamp(x, -T, T) / T       # TODO: this may cause errors ?!
        # x = torch.round(x * n) / n

        x = 2 * x - 1   # [-1, 1]
        return x  

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()
        loss_all = 0

        self.pred, self.pred_features = self.netCLS.module.forward_features(self.img)

        with torch.no_grad():
            soft_features = self.netTeach.module.forward_backbone(self.img_teach)
            soft_features = self.quant(soft_features)

        if self.losses.get("l_cls"):
            loss_cls = self.losses.get("l_cls")(self.pred, self.target)
            loss_dict["l_cls"] = loss_cls.item()
            loss_all += loss_cls * self.loss_weights["l_cls"]

        if self.losses.get("l_teach"):
            loss_teach = self.losses.get("l_teach")(self.pred_features, soft_features.detach())
            loss_dict["l_teach"] = loss_teach.item()
            loss_all += loss_teach * self.loss_weights["l_teach"]

        self.set_optimizer(names=["netCLS"], operation="zero_grad")
        loss_all.backward()
        # self.clip_grad_norm(["netCLS"], self.max_grad_norm)
        self.set_optimizer(names=["netCLS"], operation="step")

        self.log_dict = loss_dict

    def test(self, test_data, crop_size=None):
        self.img = test_data["img"].to(self.device)
        # if self.opt["pre_upsample"] is not None:
            # size = (self.opt["pre_upsample"], self.opt["pre_upsample"])
            # self.img = F.interpolate(self.img, size, mode="bicubic", align_corners=True)
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
    
    