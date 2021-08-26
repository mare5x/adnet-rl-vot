# Mostly from: https://github.com/hyeonseobnam/py-MDNet

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
import pathlib

from pytracking.features.preprocessing import numpy_to_torch
from ltr.data.image_loader import opencv_loader as image_loader


def crop_image(img, bbox, img_size=107, padding=16, flip=False, rotate_limit=0, blur_limit=0):
    x, y, w, h = np.array(bbox, dtype='float32')

    cx, cy = x + w/2, y + h/2

    if padding > 0:
        w += 2 * padding * w/img_size
        h += 2 * padding * h/img_size

    # List of transformation matrices
    matrices = []

    # Translation matrix to move patch center to origin
    translation_matrix = np.asarray([[1, 0, -cx],
                                     [0, 1, -cy],
                                     [0, 0, 1]], dtype=np.float32)
    matrices.append(translation_matrix)

    # Scaling matrix according to image size
    scaling_matrix = np.asarray([[img_size / w, 0, 0],
                                 [0, img_size / h, 0],
                                 [0, 0, 1]], dtype=np.float32)
    matrices.append(scaling_matrix)

    # Define flip matrix
    if flip and np.random.binomial(1, 0.5):
        flip_matrix = np.eye(3, dtype=np.float32)
        flip_matrix[0, 0] = -1
        matrices.append(flip_matrix)

    # Define rotation matrix
    if rotate_limit and np.random.binomial(1, 0.5):
        angle = np.random.uniform(-rotate_limit, rotate_limit)
        alpha = np.cos(np.deg2rad(angle))
        beta = np.sin(np.deg2rad(angle))
        rotation_matrix = np.asarray([[alpha, -beta, 0],
                                      [beta, alpha, 0],
                                      [0, 0, 1]], dtype=np.float32)
        matrices.append(rotation_matrix)

    # Translation matrix to move patch center from origin
    revert_t_matrix = np.asarray([[1, 0, img_size / 2],
                                  [0, 1, img_size / 2],
                                  [0, 0, 1]], dtype=np.float32)
    matrices.append(revert_t_matrix)

    # Aggregate all transformation matrices
    matrix = np.eye(3)
    for m_ in matrices:
        matrix = np.matmul(m_, matrix)

    # Warp image, padded value is set to 128  (TODO maybe border replicate?)
    patch = cv.warpPerspective(img,
                                matrix,
                                (img_size, img_size),
                                borderValue=128)

    if blur_limit and np.random.binomial(1, 0.5):
        blur_size = np.random.choice(np.arange(1, blur_limit + 1, 2))
        patch = cv.GaussianBlur(patch, (blur_size, blur_size), 0)

    return patch



def extract_region(image, bbox, crop_size=107, padding=16, means=128):
    region = crop_image(image, bbox, crop_size, padding)
    return numpy_to_torch(region - means)


# Helper class for batching transformed samples from an image.
class RegionExtractor():
    def __init__(self, image, samples, params):
        self.image = image
        self.samples = samples

        self.crop_size = params.img_size
        self.padding = params.padding
        self.means = params.means

        self.batch_size = params.batch_test

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __call__(self):
        regions = self.extract_regions(self.index)
        regions = torch.from_numpy(regions)
        return regions

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)
        regions = regions.astype('float32') - self.means  # Subtract means (based on required VGG-M input)
        regions = regions.transpose(0, 3, 1, 2)  # (N, h, w, 3) to (N, 3, h, w)
        return regions


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def gen_samples(type_, bb, n, img_size, trans_f, scale_f):    
    # bb: target bbox (min_x,min_y,w,h)
    x, y, w, h = bb

    # (center_x, center_y, w, h)
    sample = np.array([x + w / 2, y + h / 2, w, h])
    samples = np.tile(sample[None, :], (n, 1))

    # vary aspect ratio
    # if self.aspect is not None:
    #     ratio = np.random.rand(n, 2) * 2 - 1
    #     samples[:, 2:] *= self.aspect ** ratio

    if type_ == 'gaussian':
        samples[:, :2] += trans_f * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
        samples[:, 2:] *= scale_f ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)
    elif type_ == 'uniform':
        samples[:, :2] += trans_f * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
        samples[:, 2:] *= scale_f ** (np.random.rand(n, 1) * 2 - 1)
    elif type_ == 'whole':
        m = int(2 * np.sqrt(n))
        xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
        xy = np.random.permutation(xy)[:n]
        samples[:, :2] = bb[2:] / 2 + xy * (img_size - bb[2:] / 2 - 1)
        samples[:, 2:] *= scale_f ** (np.random.rand(n, 1) * 2 - 1)

    # adjust bbox range
    samples[:, 2:] = np.clip(samples[:, 2:], 10, img_size - 10)
    samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, img_size - samples[:, 2:] / 2 - 1)
    
    # (min_x, min_y, w, h)
    samples[:, :2] -= samples[:, 2:] / 2

    return samples


class SampleGenerator():
    def __init__(self, type_, img_size, trans=1, scale=1):
        self.type = type_
        self.img_size = img_size
        self.trans = trans
        self.scale = scale

    def _gen_samples(self, bb, n):
        return gen_samples(self.type, bb, n, self.img_size, self.trans, self.scale)

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):
        if overlap_range is None and scale_range is None:
            return self._gen_samples(bbox, n)
        else:
            samples = None
            remain = n
            factor = 2
            while remain > 0 and factor < 16:
                samples_ = self._gen_samples(bbox, remain * factor)

                idx = np.ones(len(samples_), dtype=bool)
                if overlap_range is not None:
                    r = overlap_ratio(samples_, bbox)
                    idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                if scale_range is not None:
                    s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                samples_ = samples_[idx, :]
                samples_ = samples_[:min(remain, len(samples_))]
                if samples is None:
                    samples = samples_
                else:
                    samples = np.concatenate([samples, samples_])
                remain = n - len(samples)
                factor = factor * 2

            return samples    


def image_threshold_mask(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Otsu's thresholding after Gaussian filtering
    # blur = cv.GaussianBlur(img, (5,5), 0)
    # _, mask = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, mask = cv.threshold(img, 35, 255, cv.THRESH_BINARY)

    # Morphology to clean up
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # # anti-alias the mask -- blur then stretch
    # # blur alpha channel
    # mask = cv.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv.BORDER_DEFAULT)

    # # linear stretch so that 127.5 goes to 0, but 255 stays 255
    # mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    return mask


def plot_image(image, boxes=[], gt=None, colormap=True):
    dpi = 80.0
    figsize = (image.shape[1] / dpi, image.shape[0] / dpi)

    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(image, aspect='auto')

    if gt is not None:
        gt_rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                                linewidth=1, edgecolor="#00ff00", zorder=1, fill=False)
        ax.add_patch(gt_rect)
    
    for i, bb in enumerate(boxes):
        color = plt.get_cmap('viridis')(i / (len(boxes) - 1)) if colormap else "#ff0000"
        rect = plt.Rectangle(tuple(bb[:2]), bb[2], bb[3],
                            linewidth=1, edgecolor=color, zorder=1, fill=False)
        ax.add_patch(rect)

    # plt.show()
    return fig, ax


def PIL_plot_image(image, boxes=[], gt=None, colormap=True):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = Image.open(image)
    draw = ImageDraw.Draw(img)

    if gt is not None:
        draw.rectangle([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]], outline='green', width=3)
    
    for i, bb in enumerate(boxes):
        color = tuple([int(x * 255) for x in plt.get_cmap('plasma')(i / (len(boxes) - 1))]) if colormap else "#ff0000"
        draw.rectangle([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]], outline=color, width=2)

    return img


def plot_images(images, gt=None, boxes=None):
    fig, axs = plt.subplots(1, len(images))    
    for i in range(len(images)):
        ax = axs[i]
        ax.imshow(images[i])
        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[i][:2]), gt[i][2], gt[i][3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        if boxes is not None:
            for bb in boxes[i]:
                rect = plt.Rectangle(tuple(bb[:2]), bb[2], bb[3],
                                    linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
                ax.add_patch(rect)
    return fig, axs


def rolling_average(x, last_k=10):
    # avg = np.convolve(x, np.ones(last_k), 'same') / last_k
    avg = np.zeros_like(x)
    for i in range(len(avg)):
        avg[i] = x[max(0, i - last_k + 1):i + 1].mean()
    return avg


def plot_graph_avg(x, last_k=10, title=None):
    fig = plt.figure(None)
    # plt.tight_layout()
    plt.cla()
    plt.plot(x)
    
    if last_k is not None:
        # avg = np.convolve(x, np.ones(last_k), 'same') / last_k
        avg = np.zeros_like(x)
        for i in range(len(avg)):
            avg[i] = x[max(0, i - last_k + 1):i + 1].mean()
        plt.plot(avg)
    
    if title is not None:
        plt.title(title)
    plt.show()


def plot_hist(x, bins=11, range_=(0, 11), title=None):
    fig = plt.figure(None)
    # plt.tight_layout()
    plt.cla()
    plt.hist(x, bins=bins, range=range_)
    if title is not None:
        plt.title(title)
    plt.show()


# Visualize optimal tracker - sanity check.
def run_optimal_tracker(sequence, params, save_path=pathlib.Path("./plots/")):
    from .train import generate_action_labels, perform_action

    save_path.mkdir(exist_ok=True)

    curr_bbox = sequence.ground_truth_rect[0]

    fig = PIL_plot_image(sequence.frames[0], gt=curr_bbox)
    fig.save(pathlib.Path(save_path) / f"{0:06}.jpg")

    for frame_num, frame_path in enumerate(sequence.frames[1:], start=1):
        image = image_loader(frame_path)

        gt = sequence.ground_truth_rect[frame_num]
        bboxes = [curr_bbox]
        round_bboxes = set([tuple(curr_bbox.round())])
        while len(bboxes) < 20:
            action = generate_action_labels(gt, curr_bbox[np.newaxis], params)
            bbox = perform_action(curr_bbox, action, params)
            next_bbox_round = tuple(bbox.round())
            if next_bbox_round in round_bboxes:
                action = params.opts['stop_action']
            round_bboxes.add(next_bbox_round)
            bboxes.append(bbox)
            curr_bbox = bbox
            if action == params.opts['stop_action']:
                break

        fig = PIL_plot_image(frame_path, bboxes, gt=gt)
        fig.save(pathlib.Path(save_path) / f"{frame_num:06}.jpg")

    print(save_path)