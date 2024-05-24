import numpy as np
import time
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchxrayvision as xrv
from enum import Enum
from PIL import Image
from skimage import exposure
from skimage.measure import label, regionprops


class XRayNormalizationType(Enum):
    IODICE = 1
    HEART_CROP = 2
    HEART_CROP_2 = 3


def _get_largest_connected_component(image):
    labels = label(image.numpy())
    assert(labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
    
    
def _convert_np_to_uint8(img):
    target_type_min = 0
    target_type_max = 255
    target_type = np.uint8
    
    imin = img.min()
    imax = img.max()
    
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)

    return new_img


def normalize_xray(img, normalization_type: XRayNormalizationType):
    mean = 0.5024
    std = 0.2898
    if normalization_type == XRayNormalizationType.IODICE:
        img_size = 1248
        crop = 1024
        
        out = exposure.equalize_hist(img.squeeze(0).numpy())
        out = _convert_np_to_uint8(out)
        out = Image.fromarray(out)
        
        time.sleep(0.1)

        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return transforms(out)
    elif normalization_type == XRayNormalizationType.HEART_CROP:
        seg_model = xrv.baseline_models.chestx_det.PSPNet()
        
        minimum = img.min()
        maximum = img.max()
        offset = (maximum + minimum) / 2
        ratio = (maximum - minimum) / 2
        xray = (img - offset) / ratio
        norm = T.Normalize(xray.mean(), xray.std())
        xray = norm(xray)

        xray = xray * 1024
        xray = xray.detach().numpy()
        transform = T.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
        xray = transform(xray)
        xray = torch.from_numpy(xray)

        with torch.no_grad():
            pred = seg_model(xray)
        
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.4] = 0
        pred[pred >= 0.4] = 1

        mask = pred[0, 8] # 8th channel is the heart segmentation
        largestCC = _get_largest_connected_component(mask)
        props = regionprops(largestCC.astype(np.uint8))
        bbox = [i for i in props[0]['bbox']]

        _, y, x = img.shape
        min_size = np.min([y, x])
        scale_ratio = min_size / 512
        if y > x:
            v_transpose = (y-x) // 2
            h_transpose = 0
        else:
            v_transpose = 0
            h_transpose = (x-y) // 2

        bbox[0] = round(bbox[0]*scale_ratio) + v_transpose
        bbox[1] = round(bbox[1]*scale_ratio) + h_transpose
        bbox[2] = round(bbox[2]*scale_ratio) + v_transpose
        bbox[3] = round(bbox[3]*scale_ratio) + h_transpose

        out = exposure.equalize_hist(img.squeeze(0).numpy())
        out = _convert_np_to_uint8(out)
        out = Image.fromarray(out)
        out = F.crop(out, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
        
        time.sleep(0.1)

        transforms = T.Compose([
            T.Resize((800, 1200)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return transforms(out)
    elif normalization_type == XRayNormalizationType.HEART_CROP_2:
        seg_model = xrv.baseline_models.chestx_det.PSPNet()
        
        minimum = img.min()
        maximum = img.max()
        offset = (maximum + minimum) / 2
        ratio = (maximum - minimum) / 2
        xray = (img - offset) / ratio
        norm = T.Normalize(xray.mean(), xray.std())
        normalized = norm(xray)

        xray = normalized * 1024
        xray = xray.detach().numpy()
        transform = T.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
        xray = transform(xray)
        xray = torch.from_numpy(xray)

        with torch.no_grad():
            pred = seg_model(xray)
        
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.4] = 0
        pred[pred >= 0.4] = 1

        mask = pred[0, 8] # 8th channel is the heart segmentation
        largestCC = _get_largest_connected_component(mask)
        props = regionprops(largestCC.astype(np.uint8))
        bbox = [i for i in props[0]['bbox']]

        _, y, x = img.shape
        min_size = np.min([y, x])
        scale_ratio = min_size / 512
        if y > x:
            v_transpose = (y-x) // 2
            h_transpose = 0
        else:
            v_transpose = 0
            h_transpose = (x-y) // 2
        
        margin = 100

        bbox[0] = round(bbox[0]*scale_ratio) + v_transpose - margin
        bbox[1] = round(bbox[1]*scale_ratio) + h_transpose - margin
        bbox[2] = round(bbox[2]*scale_ratio) + v_transpose + margin
        bbox[3] = round(bbox[3]*scale_ratio) + h_transpose + margin

        out = F.crop(normalized, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
        transforms = T.Compose([
            T.Resize((800, 1200))
        ])
        return transforms(out)
    else:
        raise Exception(f'Invalid normalization type: "{normalization_type}"')
