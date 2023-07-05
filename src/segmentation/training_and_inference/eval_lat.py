import argparse
from scipy import ndimage
import numpy as np
import json

import torch
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm as tq
import os
from math import ceil
from PIL import Image as PILImage
import cv2
from libs.datasets.latent_city import Cityscapes

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './data/cityscapes/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
INPUT_SIZE = 832
RESTORE_FROM = './deeplab_resnet.pth'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--img_dir", type=str, default="./data",help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--lbl_dir", type=str, default="./data",help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data_set", type=str, default="cityscapes", help="dataset to train")
    parser.add_argument("--arch",type=str,default="CascadeRelatioNet_res50")
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num_classes", type=int, default=34,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore models parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--tile_size_h", type=int, default=1024,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--tile_size_w", type=int, default=2048,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=True,
                        help="use whole input size.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="output dir of prediction")
    parser.add_argument("--rgb", type=str2bool, default='False')
    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, tile_size, classes, flip_evaluation):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1.0 / 3.0

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_img = torch.from_numpy(padded_img)
            padded_img = padded_img.cuda()
            padded_prediction = net(padded_img)
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def predict_whole(net, image, tile_size):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    prediction = net(image.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    # print(prediction.shape)
    return prediction


def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    # print(tile_size)
    full_probs = np.zeros((tile_size[0], tile_size[1], classes))
    for scale in scales:
        scale = float(scale)
        # print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        scaled_probs = predict_whole(net, scale_image, tile_size)
        if flip_evaluation == True:
            flip_scaled_probs = predict_whole(net, scale_image[:, :, :, ::-1].copy(), tile_size)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
        # print(full_probs.shape)
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def val():
    """Create the models and start the evaluation process."""
    args = get_arguments()

    # gpu0 = args.gpu
    # os.environ["CUDA_VISIBLE_DE VICES"] = args.gpu
    h, w = args.input_size, args.input_size
    if args.whole:
        input_size = (args.tile_size_h, args.tile_size_w)
    else:
        input_size = (h, w)
    
    # from mymodel import mydec_vic
    # model = mydec_vic.__dict__[args.arch](num_classes=args.num_classes,in_chn = args.in_chn)
    from mymodel import mydec
    model = mydec.__dict__[args.arch](num_classes=args.num_classes)
    # model = models.__dict__[args.arch](num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict,strict=False)

    model.eval()
    model.cuda()
    if args.rgb == 1:
        IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    else:
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        IMG_VARS = np.array((1, 1, 1), dtype=np.float32)

    dataset = Cityscapes(args.img_dir, args.lbl_dir, crop_size=(args.tile_size_h, args.tile_size_w), mean=IMG_MEAN, vars=IMG_VARS,
                        scale=False, mirror=False, RGB=args.rgb,eval=True)
    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    palette = get_palette(256)
    interp = nn.Upsample(size=(args.tile_size_h, args.tile_size_w), mode='bilinear', align_corners=True)

    output_preds = os.path.join(args.output_dir, "./preds")
    output_gts = os.path.join(args.output_dir,"./gts")
    output_preds_gry = os.path.join(args.output_dir, "./preds_gry")
    output_gts_gry = os.path.join(args.output_dir, "./gts_gry")
    output_results = os.path.join(args.output_dir, "./result")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(output_preds):
        os.makedirs(output_preds)
    if not os.path.exists(output_gts):
        os.makedirs(output_gts)
    if not os.path.exists(output_gts_gry):
        os.makedirs(output_gts_gry)
    if not os.path.exists(output_preds_gry):
        os.makedirs(output_preds_gry)
    if not os.path.exists(output_results):
        os.makedirs(output_results)

    for index, batch in enumerate(tq(testloader)):
        if index % 100 == 0:
            print('%d processd' % (index))
        image, label, name = batch
        with torch.no_grad():
            if args.whole:
                output = predict_multiscale(model, image, input_size, [1.0], args.num_classes, False)
            else:
                output = predict_sliding(model, image.numpy(), input_size, args.num_classes, True)

        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_im = PILImage.fromarray(seg_pred)
        output_im.putpalette(palette)
        output_im.save(os.path.join(output_preds,name[0])+".png")
        seg_gt1 = np.asarray(label[0].numpy(), dtype=np.uint8)
        output_gt = PILImage.fromarray(seg_gt1)
        output_gt.putpalette(palette)
        output_gt.save(os.path.join(output_gts,name[0])+".png")
        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        # print(np.unique(seg_pred))
        # cv2.imwrite(os.path.join(output_gts,str(index))+".png",seg_gt)
        cv2.imwrite(os.path.join(output_preds_gry,name[0])+".png",seg_pred)
        seg_gt = np.asarray(label[0].numpy(), dtype=np.int)
        cv2.imwrite(os.path.join(output_gts_gry,name[0])+".png",seg_gt)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    print({'meanIU': mean_IU, 'IU_array': IU_array})
    with open(os.path.join(args.output_dir, "result", 'result.txt'), 'w') as f:
        f.write(json.dumps({'meanIU': mean_IU, 'IU_array': IU_array.tolist()}))


if __name__ == '__main__':
    val()