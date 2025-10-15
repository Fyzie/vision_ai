#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch
import torch.backends.cudnn as cudnn
import glob
import cv2
import time

from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from data.config import update_config
from utils.output_utils import NMS, after_nms, draw_img

def main(
    # ------------------- User parameters -------------------
    trained_model = 'weights/yolact_base_54_800000.pth',  
    # Path to the trained Yolact model weights

    img_size = 640,

    traditional_nms = False,  
    # Whether to use traditional NMS instead of Yolact’s fast NMS

    hide_mask = False,  
    # If True, masks will not be drawn on the output images/videos

    hide_bbox = False,  
    # If True, bounding boxes will not be drawn on the output images/videos

    hide_score = False,  
    # If True, class scores will not be displayed on the output

    cutout = False,  
    # If True, each detected object will be cut out from the original image

    show_lincomb = False,  
    # If True, shows the mask generation process (used for visualization/debugging)

    no_crop = False,  
    # If True, output masks will not be cropped to their predicted bounding boxes

    image_folder = 'images',  
    # Path to folder containing images to run detection on. Set to None if not used.

    video_path = None,  
    # Path to a video file to run detection on. Set to None if not used.  
    # Pass a webcam index (e.g., '0') to use a webcam feed

    real_time = False,  
    # If True, will display results in real-time using cv2.imshow

    visual_thre = 0.3,  
    # Score threshold: detections with a confidence below this value will be ignored
    # -------------------------------------------------------
    ):  
    strs = trained_model.split('_')
    config = f'{strs[-3]}_{strs[-2]}_config'
    update_config(config, img_size=640)
    print(f'\nUsing \'{config}\' according to the trained_model.\n')

    with torch.no_grad():
        cuda = torch.cuda.is_available()
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        net.load_weights(trained_model, cuda)
        net.eval()
        print('Model loaded.\n')

        if cuda:
            net = net.cuda()

        # ----------------- Detect Images -----------------
        if image_folder is not None:
            images = glob.glob(os.path.join(image_folder, '*.jpg'))

            for i, one_img in enumerate(images):
                img_name = os.path.basename(one_img)
                print(f"Img Name: {img_name}")
                img_origin = cv2.imread(one_img)
                print(img_origin.shape)

                img_tensor = torch.from_numpy(img_origin).float()
                if cuda:
                    img_tensor = img_tensor.cuda()
                img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]

                img_trans = FastBaseTransform()(img_tensor.unsqueeze(0))
                net_outs = net(img_trans)
                nms_outs = NMS(net_outs, traditional_nms)

                results = after_nms(
                    nms_outs, img_h, img_w,
                    show_lincomb=show_lincomb,
                    crop_masks=not no_crop,
                    visual_thre=visual_thre,
                    img_name=img_name
                )

                img_numpy = draw_img(results, img_origin, img_name, hide_mask=hide_mask, hide_bbox=hide_bbox,
                                     hide_score=hide_score, cutout=cutout)
                print(f"Img Np: {img_numpy.shape}")

                os.makedirs('results/images', exist_ok=True)
                cv2.imwrite(f'results/images/{img_name}', img_numpy)
                print(f'\r{i + 1}/{len(images)}', end='')

            print('\nDone.')

        # ----------------- Detect Video -----------------
        elif video_path is not None:
            vid = cv2.VideoCapture(os.path.join('videos', video_path))

            target_fps = round(vid.get(cv2.CAP_PROP_FPS))
            frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

            name = os.path.basename(video_path)
            os.makedirs('results/videos', exist_ok=True)
            video_writer = cv2.VideoWriter(
                f'results/videos/{name}',
                cv2.VideoWriter_fourcc(*"mp4v"),
                target_fps,
                (frame_width, frame_height)
            )

            frame_times = MovingAverage()
            progress_bar = ProgressBar(40, num_frames)

            time_here = 0
            fps = 0
            for i in range(num_frames):
                ret, frame_np = vid.read()
                if not ret:
                    break

                frame_origin = torch.from_numpy(frame_np).float()
                if cuda:
                    frame_origin = frame_origin.cuda()

                img_h, img_w = frame_origin.shape[0], frame_origin.shape[1]
                frame_trans = FastBaseTransform()(frame_origin.unsqueeze(0))
                net_outs = net(frame_trans)
                nms_outs = NMS(net_outs, traditional_nms)

                results = after_nms(
                    nms_outs, img_h, img_w,
                    crop_masks=not no_crop,
                    visual_thre=visual_thre
                )

                if cuda:
                    torch.cuda.synchronize()

                temp = time_here
                time_here = time.time()
                if i > 0:
                    frame_times.add(time_here - temp)
                    fps = 1 / frame_times.get_avg()

                frame_numpy = draw_img(results, frame_origin, None, fps=fps,
                                       hide_mask=hide_mask, hide_bbox=hide_bbox,
                                       hide_score=hide_score, cutout=cutout)

                if real_time:
                    cv2.imshow('Detection', frame_numpy)
                    cv2.waitKey(1)
                else:
                    video_writer.write(frame_numpy)

                progress = (i + 1) / num_frames * 100
                progress_bar.set_val(i + 1)
                print(f'\rDetecting: {repr(progress_bar)} {i + 1} / {num_frames} ({progress:.2f}%) {fps:.2f} fps', end='')

            if not real_time:
                print(f'\n\nDone, saved in: results/videos/{name}')

            vid.release()
            video_writer.release()


if __name__ == "__main__":
    main(
        trained_model = r'D:\Pytorch Projects\computer_vision\training\results\Gasket\run4\weights\best_14.52_res101_custom_8244.pth',
        img_size = 640,
        traditional_nms = False,
        hide_mask = False,
        hide_bbox = False,
        hide_score = False,
        cutout = False,
        show_lincomb = False,
        no_crop = False,
        image_folder = r'D:\Pytorch Projects\computer_vision\training\vision_ai\yolact\data\gasket',  # set to your folder of images
        video_path = None,        # set to your video filename or None
        real_time = False,
        visual_thre = 0.3,
        )
