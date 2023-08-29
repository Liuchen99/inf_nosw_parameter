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
class QKDModel(BaseModel):
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
            
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.iter_CoStudy = train_opt["iter_CoStudy"]
            self.iter_Tutoring = train_opt["iter_Tutoring"]
    
            self.temperature = train_opt["temperature"]
        
    def feed_data(self, data):
        self.img = data["img"].to(self.device)
        self.target = data["target"].to(self.device)
    
    def quant(self, x, bit=3):
        Tmax = torch.max(x).detach()
        Tmin = torch.min(x).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, -T, T) / T       # TODO: this may cause errors ?!
        n = float(2 ** bit -1)
        activations_q = torch.round(x * n) / n
        # print("quant_feature:", activations_q.max(), activations_q.min())
        return activations_q

    def optimize_CoStudy(self, loss_dict):
        self.pred = self.netCLS(self.img)
        soft_target = self.netTeach(self.img)


        # update teacher
        loss_T = 0
        loss_T_cls = self.losses.get("l_cls")(soft_target, self.target)
        loss_dict["loss_T_cls"] = loss_T_cls.item()
        loss_T += loss_T_cls * self.loss_weights["l_cls"]

        loss_T_kl = self.losses.get("l_teach")(soft_target, self.pred, self.temperature)
        loss_dict["loss_T_teach"] = loss_T_kl.item()
        loss_T += loss_T_cls * self.loss_weights["l_teach"]

        self.set_optimizer(names=["netTeach"], operation="zero_grad")
        loss_T.backward()
        self.set_optimizer(names=["netTeach"], operation="step")

        # update student
        loss_S = 0
        loss_S_cls = self.losses.get("l_cls")(self.pred, self.target)
        loss_dict["loss_S_cls"] = loss_S_cls.item()
        loss_S += loss_S_cls * self.loss_weights["l_cls"]

        loss_S_kl = self.losses.get("l_teach")(self.pred, soft_target, self.temperature)
        loss_dict["loss_S_teach"] = loss_S_kl.item()
        loss_S += loss_S_cls * self.loss_weights["l_teach"]

        self.set_optimizer(names=["netCLS"], operation="zero_grad")
        loss_S.backward()
        self.set_optimizer(names=["netCLS"], operation="step")

    def optimize_Tutoring(self, loss_dict):
        loss_all = 0

        self.pred = self.netCLS(self.img)
        with torch.no_grad():
            soft_target = self.netTeach(self.img)

        # hard target loss
        if self.losses.get("l_cls"):
            loss_cls = self.losses.get("l_cls")(self.pred, self.target)
            loss_dict["l_cls"] = loss_cls.item()
            loss_all += loss_cls * self.loss_weights["l_cls"]

        # soft target loss
        if self.losses.get("l_teach"):
            loss_teach = self.losses.get("l_teach")(self.pred, soft_target.detach(), self.temperature)
            loss_dict["l_teach"] = loss_teach.item()
            loss_all += loss_teach * self.loss_weights["l_teach"]

        self.set_optimizer(names=["netCLS"], operation="zero_grad")
        loss_all.backward()
        self.set_optimizer(names=["netCLS"], operation="step")
        
        return loss_dict

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        if step < self.iter_CoStudy:
            self.set_network_state(["netTeach"], "train")
            self.set_requires_grad(["netTeach"], True)
            self.optimize_CoStudy(loss_dict)
        else:
            self.set_network_state(["netTeach"], "eval")
            self.set_requires_grad(["netTeach"], False)
            self.optimize_Tutoring(loss_dict)

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
    
    