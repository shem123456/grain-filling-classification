import inspect
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import time

# from thop import profile

import torch
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
from tqdm import tqdm
# from vit_ghost import gr8t_g100_pd8
import timm
# from tracker.gpu_mem_track import MemTracker
import pandas as pd

def main(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('{}'.format(model_name), pretrained=False, num_classes=12).to(device)
    model_weight_path = "{}/best-model.pth".format(model_name)
    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint)
    time_res = []
    pred_res = []
    non_2 = []
    non_5 = []
    pre_error = {}
    for folder in range(2, 13):
        img_list = os.listdir("datasets/test/" + str(folder*3))
        for i in tqdm(range(0, len(img_list))):
            data_transform =  transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            # load image
            img_path = "datasets/test/" + str(folder*3) + "/" + img_list[i]
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # read class_indict
            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)
      
            model.eval()
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cuda()
                predict0 = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict0).cpu().numpy()
            if int(class_indict[str(predict_cla)]) != int(folder):
                pre_error.update({img_list[i]:[int(folder*3) , int(class_indict[str(predict_cla)])]})
            pred_res.append(class_indict[str(predict_cla)])

    # print(pre_error)
    return pre_error

if __name__ == '__main__':
    model_name = 'densenet121' # 唯一要改的超参数
    pre_error = main(model_name)
    df = []
    for key,value in pre_error.items():
        image_name = key
        true_value = value[0]
        pred_value = value[1]
        line = {'image_name':image_name,'true_value':true_value,'pred_value':pred_value}
        print(line)
        df.append(line)
    df = pd.DataFrame(df)
    df.to_csv('results-{}.csv'.format(model_name),index=False)
