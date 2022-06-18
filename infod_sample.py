import numpy as np 
import json
import os
import sys
import time
import math
import io
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models  
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
from  torchattacks.attack import Attack  
import torchattacks
from utils import *
from compression import *
from decompression import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class InfoDrop(Attack):
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`. 
        
    """
    def __init__(self, model, height = 224, width = 224,  steps=40, batch_size = 20, block_size = 8, q_size = 10, targeted = False):
        super(InfoDrop, self).__init__("InfoDrop", model)
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]
        # Differential quantization
        self.alpha_range = [float(np.log(1e-9)),float(np.log(1e-20))]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size) 
        q_ini_table = np.empty((batch_size,int(block_n),block_size,block_size), dtype = np.float32)
        q_ini_table.fill(40)
        self.q_tables = {"y": torch.from_numpy(q_ini_table),
                        "cb": torch.from_numpy(q_ini_table),
                        "cr": torch.from_numpy(q_ini_table)}        
    
     
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        images = images.clone().detach().to(self.device)*255
        labels = labels.clone().detach().to(self.device)
        adv_loss =  nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([self.q_tables["y"],  self.q_tables["cb"], self.q_tables["cr"]], lr= 0.01)
        images = images.permute(0, 2, 3, 1)
        components = {'y': images[:,:,:,0], 'cb': images[:,:,:,1], 'cr': images[:,:,:,2]}
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], torch.exp(self.alpha))
                comp = dequantize(comp, self.q_tables[k]) 
                comp = idct_8x8(comp)
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp

            rgb_images = torch.cat([upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            outputs = self.model(rgb_images/255)
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels).sum()/self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels).sum()/self.batch_size).cpu().detach().numpy()


            adv_cost = adv_loss(outputs, labels) 
            
            if not self.targeted:
                adv_cost = -1* adv_cost

            total_cost = adv_cost 
            optimizer.zero_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval
            
            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() -  torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            #print(self.q_tables)
            if i%10 == 0:     
            
                print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate )
            if suc_rate >= 1:
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0)
                return q_images/255        
        q_images = torch.clamp(rgb_images, min=0, max=255.0)
       
        return q_images/255
class OriInfoDrop(Attack):
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`. 
        
    """
    def __init__(self, model, height = 224, width = 224,  steps=40, batch_size = 20, block_size = 8, q_size = 10, targeted = False):
        super(OriInfoDrop, self).__init__("OriInfoDrop", model)
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]
        # Differential quantization
        self.alpha_range = [0.1,1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size) 
        q_ini_table = np.empty((batch_size,int(block_n),block_size,block_size), dtype = np.float32)
        q_ini_table.fill(q_size)
        self.q_tables = {"y": torch.from_numpy(q_ini_table),
                        "cb": torch.from_numpy(q_ini_table),
                        "cr": torch.from_numpy(q_ini_table)}        
    
     
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        images = images.clone().detach().to(self.device)*255
        labels = labels.clone().detach().to(self.device)
        adv_loss =  nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([self.q_tables["y"],  self.q_tables["cb"], self.q_tables["cr"]], lr= 0.01)
        images = images.permute(0, 2, 3, 1)
        components = {'y': images[:,:,:,0], 'cb': images[:,:,:,1], 'cr': images[:,:,:,2]}
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], self.alpha)
                comp = dequantize(comp, self.q_tables[k]) 
                comp = idct_8x8(comp)
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp

            rgb_images = torch.cat([upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            outputs = self.model(rgb_images/255)
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels).sum()/self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels).sum()/self.batch_size).cpu().detach().numpy()


            adv_cost = adv_loss(outputs, labels) 
            
            if not self.targeted:
                adv_cost = -1* adv_cost

            total_cost = adv_cost 
            optimizer.zero_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval
            
            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() -  torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            #print(self.q_tables)
            if i%10 == 0:     
            
                print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate )
            if suc_rate >= 1:
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0)
                return q_images/255        
        q_images = torch.clamp(rgb_images, min=0, max=255.0)
       
        return q_images/255

class DCTPGD(Attack):
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`. 
        
    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            comp = block_splitting(adv_images)
            comp = dct_8x8(comp)
            comp = quantize(comp, self.q_tables[k], self.alpha)
            comp = dequantize(comp, self.q_tables[k]) 
            comp = idct_8x8(comp)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)
    
def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf =  str(round(percentage[i][index[i]].item(),2))
        pred_list.append([pred_class, pred_conf])
    return pred_list

def labels_from(net,images):
    outputs = net(images)
    _, pre = torch.max(outputs.data, 1)
    return pre
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])  
    
    
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resnet_model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
    resnet_model = resnet_model.eval()
   # target_model=
    # Uncomment if you want save results
    save_dir = "./results"
    create_dir(save_dir)
    batch_size = 40
    tar_cnt = 1000
    q_size = 40
    cur_cnt = 0
    suc_cnt = 0
    data_dir = "./test-data"
    data_clean(data_dir)
    normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)


    normal_iter = iter(normal_loader)
    pgd_suc_cnt=0
    rec_cnt=0
    rec2_cnt=0
    atk='AdvDrop'
    defense='PGD'
    defense_methods=[]
    for i in range(tar_cnt//batch_size):
        print("Iter: ", i)
        images, labels = normal_iter.next()  
        # For target attack: set random target. 
        # Comment if you set untargeted attack.
        #labels = torch.from_numpy(np.random.randint(0, 1000, size = batch_size))
        labels=labels.to(device)
        #images=images.to(device)
        defense = InfoDrop(resnet_model, batch_size=batch_size, q_size =50, steps=50, targeted = False)
        attack=InfoDrop(resnet_model, batch_size=batch_size, q_size =40, steps=50, targeted = False)
        defense2=torchattacks.PGD(resnet_model, eps=4.0/255, alpha=2.0/255, steps=5, random_start=True)
        #print(attack.device) 
        adv_images=attack(images,labels)
        wrong_labels=labels_from(resnet_model,adv_images)
        #at_images, at_labels, suc_step = attack(images, labels)
        rec_images = defense(adv_images, wrong_labels)
        rec_labels=labels_from(resnet_model,rec_images)
        rec2_images=defense2(adv_images,wrong_labels)
        rec2_labels=labels_from(resnet_model,rec2_images)
        print(labels,wrong_labels,rec_labels,rec2_labels)
        pgd_suc_cnt+=(wrong_labels != labels).sum().item()
        rec_cnt+=(rec_labels == labels).sum().item()
        rec2_cnt+=(rec2_labels == labels).sum().item()
        print("ATK suc. rate: ", pgd_suc_cnt/((i+1)*batch_size))
        print("1 rec. rate: ", rec_cnt/((i+1)*batch_size))
        print("2 rec. rate: ", rec2_cnt/((i+1)*batch_size))
        #print(pgd_images)
        # Uncomment following codes if you wang to save the adv imgs
        #print(images.device,at_images.device)
        '''
        at_images = (at_images).detach().cpu() 
        res_img = (images[0]-at_images[0]).numpy()
        pgd_images = (pgd_images).detach().cpu() 
        res_img_pgd = (images[0]-pgd_images[0]).numpy()
        adv_img=at_images[0].numpy()
        res_img = np.moveaxis(res_img, 0, 2) 
        res_img_pgd = np.moveaxis(res_img_pgd, 0, 2) 
        adv_img = np.moveaxis(adv_img, 0, 2) 
        adv_dir = os.path.join(save_dir, str(q_size))
        res_img_name = "res_{}.jpg".format(i)
        adv_img_name = "adv_{}.jpg".format(i)
        res_img_pgd_name = "res_pgd_{}.jpg".format(i)
        ori_img_name="ori_{}.jpg".format(i)
        ori_img= images[0].numpy()
        ori_img = np.moveaxis(ori_img, 0, 2) 
        save_img(ori_img,ori_img_name,adv_dir)
        save_img(adv_img, adv_img_name, adv_dir)
        save_img(res_img, res_img_name, adv_dir)
        save_img(res_img_pgd, res_img_pgd_name, adv_dir)
        '''
        #labels = labels.to(device)
        #suc_cnt += (at_labels == labels).sum().item()
        #print("Current suc. rate: ", suc_cnt/((i+1)*batch_size))
    score_list = np.zeros(tar_cnt)
    score_list[:suc_cnt] = 1.0
    stderr_dist = np.std(np.array(score_list))/np.sqrt(len(score_list))
    print('Avg suc rate: %.5f +/- %.5f'%(suc_cnt/tar_cnt,stderr_dist))
