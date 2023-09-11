import argparse
import yaml
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from utils.dataloader import DataTransform
from models.pspnet import PSPNet

DATASET_NCLASS = 21 # VOC: 21
DISR_TH = 0 # threshold for disR (dissimilarity among prediction results over patches)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/cfg_sample_segment.yaml', help='config file')
    return parser

def segment_val(cfg):
    
    #----- Network Model ------#
    net = PSPNet(n_classes=DATASET_NCLASS, padding_mode=cfg['padding_mode'])
    [dataset_name, weights_file_path] = cfg['weights']
    state_dict = torch.load(weights_file_path, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(state_dict)
    
    print('[Network model] Pretrained weights were loaded.')
    print(cfg['weights'])

    #val_data = np.load(cfg['val_images'])
    val_data = np.loadtxt(cfg['val_images'], delimiter=',', dtype=str)
    expanded_size = cfg['expanded_size']
    iou_data = []
    data_summary = [] #[fid, m_iou, m_iou_weighted, sum_effective_px, meanE, DisR]
    for fid, filename in enumerate(val_data):
    #for fid, filename in enumerate(val_data[0:3]): #debug
        print('---')
        #----- 1. load image & resize ------#
        print('{0}th image:'.format(fid))
        #filename = filename.replace('\n','') # remove return code
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
        iou_img = np.zeros((num_py * num_px, 6)) # [fid, py, px, iou, iou_weighted, effective_px]
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
                #print(x.shape)
                x = x.to(device) # Send data to GPU if possible
                outputs = net(x) # outputs = (output, output_aux, cap_loss) for CAP, (output, output_aux) for others
                y = outputs[0]
                
                y_org = y[0].to('cpu').detach().numpy().copy() #(21, 475, 475)
                y_org = np.argmax(y_org, axis=0) # (475, 475) same as 'anno_cut'
                
                # count up predicted class for each patch pixel for calculating entropy
                for category in range(DATASET_NCLASS):
                    pred_cnt[py:py+y_org.shape[0], px:px+y_org.shape[1], category][y_org==category] += 1
                
                # calculate Intersection over Union (IoU)
                y_org = y_org.reshape(y_org.shape[0], y_org.shape[1], 1)
                anno_org = anno_cut.to('cpu').detach().numpy().copy()
                labels_num = np.arange(DATASET_NCLASS).reshape(1, 1, DATASET_NCLASS)
                anno_org = anno_org.reshape(anno_org.shape[0], anno_org.shape[1], 1)

                sum_set_array = (y_org == labels_num) | (anno_org == labels_num)
                sum_set = np.sum(sum_set_array, axis=(0, 1))
                product_set_array = (y_org == labels_num) & (anno_org == labels_num)
                product_set = np.sum(product_set_array, axis=(0, 1))

                # delete background class (first class) for iou calculation
                sum_set = sum_set[1:]
                product_set = product_set[1:]

                sum_set_non_zero = sum_set[sum_set != 0]
                product_set_non_zero = product_set[sum_set != 0]
                iou_org = product_set_non_zero/sum_set_non_zero
                
                # Calculate the number of effective pixels to calculate weighted average of iou.
                # The effective pixels have union (sum_set) = 1 at least one class.
                effective_px = np.sum(np.sum(sum_set_array, axis=2) > 0)
                
                if sum_set_non_zero.size == 0:
                    iou = 0
                    iou_weighted = 0
                else:
                    iou = np.mean(iou_org)
                    iou_weighted = np.average(iou_org, weights=sum_set_non_zero) # weighted average

                # iou_img: [fid, py, px, iou, iou_weighted, effective_px]
                [yn, xn] = [int(py/patch_stride), int(px/patch_stride)]
                data_id = yn * num_px + xn
                iou_img[data_id] = [fid, py, px, iou, iou_weighted, effective_px]
        
        # calculating entroy
        def calc_entropy(cnt, exclude_background = False, annotations = None):
            # cnt: height x width x classes
            # background = 0th class
            # exclude_background = True: pixels with background class in annotations are excluded.
            BACKGROUND_CLASS = 0

            cnt = cnt.reshape(-1, cnt.shape[2]) # reshape to (pixels x classes)
            p = cnt / np.sum(cnt, axis=1).reshape(cnt.shape[0], 1) # probability
            nzi = np.where(p!=0) # nzi = (ndarray[points], ndarray[points])
            ei = np.zeros(p.shape)
            ei[nzi] = - p[nzi] * np.log2(p[nzi])
            e = np.sum(ei, axis=1) # sum ei over classes to calculate entory at each pixel
            if exclude_background is True:
                e = e[annotations.flatten() != BACKGROUND_CLASS]
            if e.size == 0: # no pixel (all pixels were judged into the backgrournd class)
                meanE = np.nan
                disR = np.nan
            else:
                meanE = np.mean(e)
                disR = np.sum(e > DISR_TH) / e.size
            return meanE, disR

        # # calculating entroy. Prediction results of background are removed.
        # # This is not good, since miss-classification into the background class is ignored in the evaluation of translation invariance. Instead, the background class is excluded based on the annotation above.
        # def calc_entropy(cnt, exclude_background = False):
        #     # cnt: height x width x classes
        #     # background = 0th class

        #     cnt = cnt.reshape(-1, cnt.shape[2]) # reshape to (pixels x classes)
        #     if exclude_background:
        #         cnt = cnt[:, 1:] # exclude background class
        #         cnt = cnt[np.where(np.sum(cnt, axis=1) != 0), :][0] # extract pixels with sum over classes is not 0
        #     if cnt.size == 0: # no pixel (all pixels were judged into the backgrournd class)
        #         meanE = np.nan
        #         disR = np.nan
        #     elif cnt.shape[0] == 1: # only 1 pixel is available
        #         # since the pixel was always classified into one class, e can not be calculated (always e=0)
        #         meanE = np.nan
        #         disR = np.nan
        #     else:
        #         p = cnt / np.sum(cnt, axis=1).reshape(cnt.shape[0], 1) # probability
        #         nzi = np.where(p!=0) # nzi = (ndarray[points], ndarray[points])
        #         ei = np.zeros(p.shape)
        #         ei[nzi] = - p[nzi] * np.log2(p[nzi])
        #         e = np.sum(ei, axis=1) # sum ei over classes to calculate entory at each pixel
        #         meanE = np.mean(e)
        #         disR = np.sum(e > DISR_TH) / e.size
        #     return meanE, disR

        # calculate entropy for evaluating translation invariance (including background class)
        cnt_cut = pred_cnt[ph:im_array.shape[0]-ph, pw:im_array.shape[1]-pw, :] # pixels with full overlapping pathces, (ph, pw) patch size
        meanE_in, disR_in = calc_entropy(cnt_cut, exclude_background=False)
        print('- translation invariance')
        print('meanE (includeing background): ' + str(meanE_in))
        print('disR (includeing background): ' + str(disR_in))

        # calculate entropy for evaluating translation invariance (excluding background class)
        cnt_cut = pred_cnt[ph:im_array.shape[0]-ph, pw:im_array.shape[1]-pw, :] # pixels with full overlapping pathces excluding background class (0th class), (ph, pw) patch size
        annotations_cut = anno_array[ph:im_array.shape[0]-ph, pw:im_array.shape[1]-pw]
        meanE_ex, disR_ex = calc_entropy(cnt_cut, exclude_background=True, annotations=annotations_cut)
        print('meanE (excluding background): ' + str(meanE_ex))
        print('disR (excluding background): ' + str(disR_ex))

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
        data_summary.append(np.array([fid, m_iou, m_iou_weighted, sum_effective_px, meanE_in, disR_in, meanE_ex, disR_ex]).reshape(1, -1))

    data_summary = np.concatenate(data_summary)
    idx = ['fid', 'm_iou', 'm_iou_weighted', 'sum_effective_px', 'meanE_in', 'disR_in', 'meanE_ex', 'disR_ex']
    pd.DataFrame(data=data_summary, columns=idx).to_csv(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'data_summary.csv')

    iou_data = np.concatenate(iou_data)
    idx = ['fid', 'py', 'px', 'iou', 'iou_weighted', 'effective_px']
    pd.DataFrame(data=iou_data, columns=idx).to_csv(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'iou_data.csv')
    
    # np.savetxt(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'data_summary.csv', data_summary, delimiter=',', fmt='%8.16f')
    # np.savetxt(cfg['outputs'] + 'segment_val' + '_' + cfg['padding_mode'] + '_' + 'iou_data.csv', iou_data, delimiter=',', fmt='%8.16f')
    
    # averaged results over all images
    m_iou = np.mean(data_summary[:,1])
    m_iou_weighted = np.average(data_summary[:, 2], weights=data_summary[:, 3])
    m_meanE_in = np.mean(data_summary[:, 4][~np.isnan(data_summary[:, 4])])
    m_disR_in = np.mean(data_summary[:, 5][~np.isnan(data_summary[:, 5])])
    m_meanE_ex = np.mean(data_summary[:, 6][~np.isnan(data_summary[:, 6])])
    m_disR_ex = np.mean(data_summary[:, 7][~np.isnan(data_summary[:, 7])])
    out = np.array([
        'padding_mode: ' + cfg['padding_mode'],
        'm_meanE (includeing background): ' + str(m_meanE_in),
        'm_disR (includeing background): ' + str(m_disR_in),
        'm_meanE (excluding background): ' + str(m_meanE_ex),
        'm_disR (excluding background): ' + str(m_disR_ex),
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
