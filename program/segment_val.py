import argparse
import yaml
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.utils

from utils.dataloader import make_datapath_list, DataTransform
from models.pspnet import PSPNet

DATASET_NCLASS = 21 # VOC: 21
DISR_TH = 0 # threshold for disR (dissimilarity among prediction results over patches)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/cfg_sample_segment.yaml', help='config file')
    return parser

def segment_val(cfg):
    
    #----- Dataset ------#
    rootpath = cfg['dataset']
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)
    
    #----- Network Model ------#
    net = PSPNet(n_classes=DATASET_NCLASS, padding_mode=cfg['padding_mode'])
    [dataset_name, weights_file_path] = cfg['weights']
    state_dict = torch.load(weights_file_path, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(state_dict)
    
    print('[Network model] Pretrained weights were loaded.')
    print(cfg['weights'])

    m_IoU_sum_total = 0
    val_data = np.load(cfg['val_images'])
    expanded_size = cfg['expanded_size']
    iou_data = []
    data_summary = [] #[fid, m_iou, m_iou_weighted, sum_effective_px, meanE, DisR]
    #for fid, filename in enumerate(val_data):
    for fid, filename in enumerate(val_data[0:3]): #debug
        print('---')
        #----- 1. load image & resize ------#
        print('{0}th image:'.format(fid))
        filename = filename.replace('\n','') # remove return code
        image_file_path = cfg['dataset'] + 'JPEGImages/' + filename + '.jpg'
        anno_file_path = cfg['dataset'] + 'SegmentationClass/' + filename + '.png'
        print(image_file_path)
        
        img = Image.open(image_file_path) # (width x height)
        anno = Image.open(anno_file_path)
        img_width, img_height = img.size
        anno_width, anno_height = anno.size
        if img_width > img_height: # resize the long side of the image to 1050 px (cfg['expanded_size'])
            img = img.resize((int((expanded_size/img_height) * img_width), expanded_size))
        else:
            img = img.resize((expanded_size, int((expanded_size/img_width) * img_height)))
        im_array = np.asarray(img)
        #plt.imsave(samples/filename + '.jpg', im_array) # to confirm the image

        if anno_width > anno_height:
            anno = anno.resize(((int((expanded_size/anno_height) * anno_width), expanded_size)), Image.NEAREST)
        else:
            anno = anno.resize(((expanded_size ,int((expanded_size/anno_width) * anno_height))), Image.NEAREST)
        anno_array = np.asarray(anno)
        #plt.imsave(samples/filename + '.png', anno_array) # to confirm the image
        
        #----- 2. Instance of Preprocessing Class ------#
        transform = DataTransform(input_size=475, color_mean=cfg['color_mean'], color_std=cfg['color_std'])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        ph = cfg['input_size'] #475
        pw = cfg['input_size'] #475
        patch_stride = cfg['patch_stride'] #47
        num_py = len(np.arange(0, im_array.shape[0]-ph+1, patch_stride))
        num_px = len(np.arange(0, im_array.shape[1]-pw+1, patch_stride))
        np.set_printoptions(precision=5, suppress=True) #default: precision=8, suppress=False
        iou_img = np.zeros((num_py * num_px, 6)) # [fid, py, px, iou, iou_weighted, sum_set_px]
        pred_cnt = np.zeros((im_array.shape[0], im_array.shape[1], DATASET_NCLASS)) # count prediction results of sliding patches
        for py in np.arange(0, im_array.shape[0]-ph+1, patch_stride):
            for px in np.arange(0, im_array.shape[1]-pw+1, patch_stride):
                im_array_cut  = im_array[py:py+ph, px:px+pw, :] # Crop an image patch with the size of (475, 475, 3), ndarray
                anno_cut = anno.crop((px, py, px+pw, py+ph)) # Crop an annotation patch with the size of (475, 475), Image
                
                #----- 3. Preprocessing ------#
                phase = "val"
                im_cut = Image.fromarray(np.uint8(im_array_cut))
                # im_cut.save(samples/filename + '_patch.jpg') # to confirm the image
                # anno_cut.save(samples/filename + '_patch.png') # to confirm the image
                im_cut, anno_cut = transform(phase, im_cut, anno_cut)
                
                # # to confirm the image
                # im_cut_path_array = im_cut.to('cpu').detach().numpy().transpose(1, 2, 0)
                # anno_cut_path_array = anno_cut.to('cpu').detach().numpy()
                # im_tmp = im_cut_path_array - np.min(im_cut_path_array)
                # im_tmp = im_tmp / np.max(im_tmp)
                # plt.imsave(samples/filename + '_patch.jpg', im_tmp) # to confirm the image
                # im_tmp = anno_cut_path_array - np.min(anno_cut_path_array)
                # im_tmp = im_tmp / np.max(im_tmp)
                # plt.imsave(samples/filename + '_patch.png', im_tmp) # to confirm the image
            
                #----- 4. Inference with PSPNet ------#
                net.eval()
                x = im_cut.unsqueeze(0) 
                outputs = net(x) # outputs = (output, output_aux, cap_loss) for CAP, (output, output_aux) for others
                y = outputs[0]
                
                y_org = y[0].to('cpu').detach().numpy().copy() #(21, 475, 475)
                y_org = np.argmax(y_org, axis=0) # (475, 475) same as 'anno_cut'
                
                # count up predicted class for each patch pixel
                for category in range(DATASET_NCLASS):
                    pred_cnt[py:py+y_org.shape[0], px:px+y_org.shape[1], category][y_org==category] += 1
                
                # calculate Intersection over Union (IoU)
                y_org = y_org.reshape(y_org.shape[0], y_org.shape[1], 1)
                anno_org = anno_cut.to('cpu').detach().numpy().copy()
                labels_num = np.arange(DATASET_NCLASS).reshape(1, 1, DATASET_NCLASS)
                anno_org = anno_org.reshape(anno_org.shape[0], anno_org.shape[1], 1)

                sum_set = np.sum((y_org == labels_num) | (anno_org == labels_num), axis=(0, 1))
                product_set = np.sum((y_org == labels_num) & (anno_org == labels_num), axis=(0, 1))
                sum_set = sum_set[1:] # delete background class (first class)
                product_set = product_set[1:]
                sum_set_non_zero = sum_set[sum_set != 0]
                product_set_non_zero = product_set[sum_set != 0]
                iou_org = product_set_non_zero/sum_set_non_zero
                
                if sum_set_non_zero.size == 0:
                    iou = 0
                    iou_weighted = 0
                    sum_set_px = 0
                else:
                    iou = np.mean(iou_org)
                    iou_weighted = np.average(iou_org, weights=sum_set_non_zero) # weighted average
                    sum_set_px = np.sum(sum_set_non_zero)

                # iou_img: [fid, py, px, iou, iou_weighted, sum_set_px]
                [yn, xn] = [int(py/patch_stride), int(px/patch_stride)]
                data_id = yn * num_px + xn
                iou_img[data_id] = [fid, py, px, iou, iou_weighted, sum_set_px]
        
        # calculate entropy for evaluating translation invariance
        # (ph, pw) patch size
        cnt_cut = pred_cnt[ph:im_array.shape[0]-ph, pw:im_array.shape[1]-pw, :]
        p = cnt_cut / np.sum(cnt_cut, axis=2).reshape(cnt_cut.shape[0], cnt_cut.shape[1], 1)
        nzi = np.where(p!=0) # nzi = (ndarray[points], ndarray[points], ndarray[points])
        #print(nzi[0].shape)
        #nz = np.array([nzi[0], nzi[1], nzi[2]]) # nz[:, 0]: index of 1st point
        ei = np.zeros(p.shape)
        ei[nzi] = - p[nzi] * np.log2(p[nzi])
        #ei = (- p * np.log2(p))*(p>0) # ei=0 when p=0, else calculate entropy
        e = np.sum(ei, axis=2) # sum ei over classes to calculate entory at each pixel
        meanE = np.mean(e)
        disR = np.sum(e > DISR_TH) / e.size
        print('- translation invariance')
        print('meanE: ' + str(meanE))
        print('disR: ' + str(disR))

        # calculate Intersection over Union (IoU) to evaluate accuracy
        iou_effective = iou_img[:,3][iou_img[:,5]>0]
        iou_weighted_effective = iou_img[:,4][iou_img[:,5]>0]
        effective_px = iou_img[:, 5][iou_img[:, 5]>0]
        m_iou = np.mean(iou_effective)
        m_iou_weighted = np.average(iou_weighted_effective, weights=effective_px)
        sum_effective_px = np.sum(effective_px)
        print('- Classification Accuracy')
        print('image id: ' + str(fid))
        print('m_iou: ' + str(m_iou))
        print('m_iou_weightd: ' + str(m_iou_weighted))
        print('effective pixels: ' + str(sum_effective_px))
        iou_data.append(iou_img)

        # data log
        data_summary.append(np.array([fid, m_iou, m_iou_weighted, sum_effective_px, meanE, disR]).reshape(1, -1))

    data_summary = np.concatenate(data_summary)
    idx = ['fid', 'm_iou', 'm_iou_weighted', 'sum_effective_px', 'meanE', 'disR']
    pd.DataFrame(data=data_summary, columns=idx).to_csv(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'data_summary.csv')

    iou_data = np.concatenate(iou_data)
    idx = ['fid', 'py', 'px', 'iou', 'iou_weighted', 'sum_set_px']
    pd.DataFrame(data=iou_data, columns=idx).to_csv(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'iou_data.csv')
    
    # np.savetxt(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'data_summary.csv', data_summary, delimiter=',', fmt='%8.16f')
    # np.savetxt(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'iou_data.csv', iou_data, delimiter=',', fmt='%8.16f')
    
    # averaged results over all images
    m_iou = np.mean(data_summary[:,1])
    m_iou_weighted = np.average(data_summary[:, 2], weights=data_summary[:, 3])
    m_meanE = np.mean(data_summary[:, 4])
    m_disR = np.mean(data_summary[:, 5])
    out = np.array([
        'paddingmode: ' + cfg['padding_mode'],
        'm_meanE: ' + str(m_meanE),
        'm_disR: ' + str(m_disR),
        'm_iou for all images: ' + str(m_iou),
        'm_iou_weightd for all images: ' + str(m_iou_weighted)
    ])
    print('------------------')
    for n in range(len(out)): print(out[n])
    np.savetxt(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'averaged_results.csv', out, fmt="%s")

if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    segment_val(cfg)
