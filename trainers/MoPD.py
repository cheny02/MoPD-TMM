import os.path as osp
import numpy as np
import torch
import math
import torch.nn as nn
import random
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .mopd_templates import CUSTOM_TEMPLATES, LASP_PROMPTS, NOISY_PROMPTS
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 
        #x=shape[100,77,512]
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #x=shape[100,512]
        return x 


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a' 
            ctx_init = temp.replace("_", " ") 
            n_ctx = len(ctx_init.split(" ")) 
            prompt = clip.tokenize(ctx_init) 
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) 

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] #torch.Size([4, 512])
            prompt_prefix = ctx_init #‘a photo of a’
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx) 
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized,torch.Size([4, 512]) ,requires_grad=True

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames] 
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        #prompt pool
        temp_1 = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        # temp_2 = LASP_PROMPTS
        # temp_3 = NOISY_PROMPTS
        temp = temp_1

        all_hard_prompt_fea=[]
        for hard_prompt in temp:
            prompts_ = [hard_prompt.format(c.replace("_", " ")) for c in classnames]
            print(f"Prompts: {prompts_}")
            prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
            prompts_ = prompts_.cuda()
            with torch.no_grad():
                text_features = clip_model_.encode_text(prompts_)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_hard_prompt_fea.append(text_features)
        all_hard_prompt_fea = torch.stack(all_hard_prompt_fea)
        self.all_hard_prompt_fea = all_hard_prompt_fea.type(dtype)

        #---------------------Moe gate
        num_allprompts = all_hard_prompt_fea.shape[0]
        gate = torch.zeros(512, num_allprompts, dtype=dtype)
        self.gate = nn.Parameter(gate, requires_grad=True)
        #-----------------------------

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION 

    def forward(self):
        ctx = self.ctx # ctx = context
        gate = self.gate
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     #[4,512]
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts , gate


class CustomCLIP(nn.Module): 
    def __init__(self, cfg, classnames, clip_model, T, num_prompts, epoch, max_epoch):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.all_hard_prompt_fea
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.T =  T
        self.num_prompts = num_prompts
        self.epoch = epoch
        self.max_epoch = max_epoch

    def forward(self, image): 
        prompts, gate = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        text_features_old = self.ori_embedding
        num_allprompts = text_features_old.shape[0]
        prob_logits = F.log_softmax(logits/self.T,dim=1) 
        clean_logits = image_features @ gate 
        top_value , top_index= clean_logits.topk(k=self.num_prompts, largest=True, dim=1)
        top_value_soft = F.softmax(top_value, dim=1) 
        text_features_old = torch.transpose(text_features_old, 1, 2)
        logits_teacher = logit_scale * image_features @ text_features_old
        prob_logits_teacher = F.softmax(logits_teacher/self.T,dim=-1) 
        prob_logits_teacher = torch.transpose(prob_logits_teacher,0,1)
        im_batch = prob_logits_teacher.shape[0]
        index_1 = torch.arange(0,im_batch,1)
        soft_loss=nn.KLDivLoss(reduction='none')
        loss_distillation = 0
        for i in range(self.num_prompts):
            prob_logits_teacher_i = prob_logits_teacher[index_1,top_index[:,i]]
            loss_distillation_1 = torch.sum((self.T**2)*soft_loss(prob_logits,prob_logits_teacher_i),dim=1)
            loss_distillation +=torch.sum(top_value_soft[:,i] * loss_distillation_1) / im_batch
        logits_teacher_T = torch.transpose(logits_teacher,0,1)
        return logits, loss_distillation, logits_teacher_T, top_value_soft, top_index, index_1

    def inference(self, image):
        prompts, _ = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()
        return logits

@TRAINER_REGISTRY.register()
class MoPD(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        epoch = self.epoch
        max_epoch = cfg.OPTIM.MAX_EPOCH
        classnames = self.dm.dataset.classnames
        # all_classnames = self.dm.dataset.all_class_names

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp": #fp32=Full Precise Float 32
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.T = cfg.LOSS.T
        self.num_prompts = cfg.LOSS.num_prompts
        self.model = CustomCLIP(cfg, classnames, clip_model, self.T, self.num_prompts, epoch, max_epoch)
        self.w = cfg.TRAINER.COOP.W
        self.w2 = cfg.TRAINER.COOP.W2

        print("Turning off gradients in both the image and the text encoder") 
        for name, param in self.model.named_parameters():
            #if "prompt_learner" not in name: :
            # if "ctx" not in name: 
            if "prompt_learner" not in name: 
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, loss_distillation, logits_teacher_T, top_value_soft, top_index, index_1 = self.model(image)

            loss_MPS = 0
            for i in range(self.num_prompts):
                logits_teacher_T_i = logits_teacher_T[index_1,top_index[:,i]]
                loss_MPS_i = F.cross_entropy(logits_teacher_T_i, label, reduction='none')
                loss_MPS +=torch.sum(top_value_soft[:,i] * loss_MPS_i) / len(index_1)

            loss_1 = F.cross_entropy(output, label)
            loss_2 = loss_distillation 
            loss = self.w*loss_1 + (1-self.w)*loss_2 + self.w2*loss_MPS

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()
            self.sched.step()
            #self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.model(input)[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model.inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

