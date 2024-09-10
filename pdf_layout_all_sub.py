import os
import cv2
import json
import yaml
import time
import pytz
import datetime
import argparse
import shutil
import torch
import numpy as np
import gc
import glob

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor
from struct_eqtable import build_model

from modules.extract_pdf import load_pdf_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor


# Category dictionary
category_dict = {
    0: 'title',
    1: 'plain text',
    2: 'abandon',
    3: 'figure',
    4: 'figure_caption',
    5: 'table',
    6: 'table_caption',
    7: 'table_footnote',
    8: 'isolate_formula',
    9: 'formula_caption',
    10:'',
    11:'',
    12:'',
    13:'inline_formula',
    14:'isolated_formula',
    15:'ocr_text'
}


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


# def save_cropped_element(image, bbox, output_dir, basename, element_type, idx):
#     xmin, ymin, xmax, ymax = bbox
#     cropped_img = image.crop((xmin, ymin, xmax, ymax))
#     element_dir = os.path.join(output_dir, element_type)
#     os.makedirs(element_dir, exist_ok=True)
#     file_name = f'{basename}_{element_type}_{xmin}_{ymin}_{xmax}_{ymax}.pdf'
#     cropped_img.save(os.path.join(element_dir, f'{basename}_{element_type}_{idx}.pdf'), 'PDF')


def save_cropped_element(image, bbox, output_dir, basename, element_type, page_idx, element_idx):
    xmin, ymin, xmax, ymax = bbox
    cropped_img = image.crop((xmin, ymin, xmax, ymax))
    element_dir = os.path.join(output_dir, element_type)
    os.makedirs(element_dir, exist_ok=True)
    file_name = f'{basename}_{element_type}_page{page_idx}_element{element_idx}.pdf'
    cropped_img.save(os.path.join(element_dir, file_name), 'PDF')
    

def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def get_image_file_list_nested(input_dir, output_dir, infer_list=None):
    img_file_list = []
    output_dirs = []

    if infer_list and not os.path.exists(infer_list):
        raise Exception("Infer list not found: {}".format(infer_list))

    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(input_dir, image_path)
            relative_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            output_dirs.add(os.path.dirname(output_path))
            img_file_list.append((image_path, output_path))
    else:
        if input_dir is None or not os.path.exists(input_dir):
            raise Exception("Input directory not found: {}".format(input_dir))

        img_endings = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.split('.')[-1].lower() in img_endings:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)
                    output_dirs.append(os.path.dirname(output_path))
                    img_file_list.append(file_path)

    if len(img_file_list) == 0:
        raise Exception("No image files found in {}".format(input_dir))
    
    img_file_list = sorted(img_file_list)
    return img_file_list, sorted(list(output_dirs))


def PDF_layout_all_sub(args, pdf, output_dir, batch_size = 128):
    args.pdf = pdf
    args.output = output_dir
    args.batch_size = batch_size
    
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Started!')

    ## ======== model init ========##
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    img_size = model_configs['model_args']['img_size']
    conf_thres = model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device = model_configs['model_args']['device']
    dpi = model_configs['model_args']['pdf_dpi']
    mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
    layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Model init done!')
    ## ======== model init ========##

    start = time.time()
    if os.path.isdir(args.pdf):
        all_pdfs, output_file_list = get_image_file_list_nested(args.pdf, args.output)
        print("image_file_list", all_pdfs)
        print("output_file_list", output_file_list)
        # all_pdfs = [os.path.join(args.pdf, name) for name in os.listdir(args.pdf)]
        # # all_pdfs = []
        # all_pdfs_second_dir_small = []
        # for second_dir_small in os.listdir(args.pdf):
        #     second_dir = os.path.join(args.pdf,second_dir_small)
        #     for pdf_path in glob.glob(f"{second_dir}/*.pdf"):
        #         pdf_name = os.path.basename(pdf_path)[:-4]
        #         all_pdfs.append(pdf_path)
        #         all_pdfs_second_dir_small.append(second_dir_small)
    elif os.path.isfile(args.pdf):
        all_pdfs = [args.pdf]
    print("total files:", len(all_pdfs))

    for idx_pdf, single_pdf in enumerate(all_pdfs):
        try:
            img_list = load_pdf_fitz(single_pdf, dpi=dpi)
        except:
            img_list = None
            print("Unexpected PDF file:", single_pdf)
        if img_list is None:
            continue
        print("pdf index:", idx_pdf, "pages:", len(img_list))
        # layout detection and element extraction
        doc_layout_result = []
        for idx, image in enumerate(img_list):
            img_H, img_W = image.shape[0], image.shape[1]
            layout_res = layout_model(image, ignore_catids=[])
            count = 0
            for res in layout_res['layout_dets']:
                xmin, ymin, xmax, ymax = res['poly'][0], res['poly'][1], res['poly'][4], res['poly'][5]
                category_id = int(res['category_id'])
                element_type = category_dict.get(int(res['category_id']), 'unknown')
                save_cropped_element(Image.fromarray(image), (xmin, ymin, xmax, ymax), output_file_list[idx_pdf], os.path.basename(single_pdf)[:-4], element_type, idx, count)
                count+=1
            
            layout_res['page_info'] = dict(
                page_no=idx,
                height=img_H,
                width=img_W
            )
            doc_layout_result.append(layout_res)

            torch.cuda.empty_cache()
            gc.collect()

        output_root = args.output
        if os.path.isdir(args.pdf):
            output_dir = output_file_list[idx_pdf]
        else:
            output_dir = output_root
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)

    now = datetime.datetime.now(tz)
    end = time.time()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('Finished! time cost:', int(end-start), 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    print(args)
    
    PDF_layout_all_sub(args, args.pdf, args.output)
