import math
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

# GPU version
class GradCAM_YOLOV3(object):
    """
    Grad CAM for Yolo V3 in mmdetection framework
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, data, index=0):
        """
        :param image: cv2 format, single image
        :param index: Which bounding box
        :return:
        """
        self.net.zero_grad()
        # Important
        feat = self.net.extract_feat(data['img'][0].cuda())
        res = self.net.bbox_head.simple_test(
            feat, data['img_metas'][0], rescale=True)
        
        score = res[0][0][index][4]
       
        score.backward()

        gradient = self.gradient  # [C,H,W]
        weight = torch.mean(gradient, axis=(2, 3))[0]  # [C]

        feature = self.feature[0]  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)  # ReLU

        # Normalization
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        # resize to 224*224
        box = res[0][0][index][:-1].cpu().detach().numpy().astype(np.int32)
        
        class_id = res[0][1][index].cpu().detach().numpy()
        return cam.cpu().detach().numpy(), box, class_id
def prepare_img(imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    return data
def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

import os
import glob
from tqdm import tqdm

config = 'configs/yolox/yolox_s_voc.py'
# config = 'configs/yolox/yolox_s_8x8_300e_coco.py'
cfg = Config.fromfile(config)
model_img_dir = rf"F:\vs_test\毕业论文\数据文件voc\idea2\CKD\dkd\fg_bg_dkd"
checkpoint = glob.glob(model_img_dir + "/stu*.pth")[0]
device = 'cuda:0'
model = init_detector(config, checkpoint, device)
grad_cam = GradCAM_YOLOV3(model, 'bbox_head.multi_level_conv_obj.2')
# bbox_head.multi_level_conv_obj.2
# bbox_head.multi_level_conv_reg.2   bbox_head.multi_level_conv_cls.2
# grad_cam = GradCAM_YOLOV3(model, 'neck.out_convs.2.conv')
img_dir = os.path.join(model_img_dir, "img")
img_list = os.listdir(img_dir)
pbar=tqdm(total=len(img_list))
for path in img_list:
    pbar.update(1)
    # if not '000021' in path:
    #     continue
    if (not path.endswith("jpg")) or "attent" in path:
        continue
    image_path = os.path.join(img_dir, path)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    data = prepare_img(image)
    ## First is the data, second is the index of the predicted bbox
    mask, box, class_id = grad_cam(data,0)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    image_cam, heatmap = gen_cam(image, mask)
    # cv2.imwrite(image_path.replace('.jpg','attent1.jpg'),image_cam)
    img_write = cv2.imencode(".jpg",image_cam)[1].tofile(image_path.replace('.jpg','attent1.jpg'))
    # img_write = cv2.imencode(".jpg",image_cam)[1].tofile(image_path.replace('.png','attent2.png'))
    # plt.imshow(image_cam[:, :, ::-1])
    # plt.show()
# plt.imshow(heatmap[:, :, ::-1])
# plt.show()
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        print(name,',',m)# backbone.conv_res_block5.res3.conv2.conv
