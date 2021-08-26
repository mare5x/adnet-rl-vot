import os
import math
import shutil
import pathlib
import itertools

from vot.dataset import VOTSequence
from vot.region import Rectangle, write_file, RegionType
from vot.utilities import write_properties
from pytracking.evaluation.votdataset import VOTDatasetWrapped

from PIL import Image, ImageFilter, ImageColor, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList

from . import utils 


class MyObjects:
    def __init__(self):
        env = env_settings()
        base_path = pathlib.Path(env.synthetic_path).with_name("objects")

        self.objs = {}
        for d in base_path.iterdir():
            if not d.is_dir():
                continue 
            self.objs[d.name] = sorted(d.glob("???.png"), key=lambda p: int(p.stem))
        
    def __getitem__(self, key):
        return self.objs[key]


# https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
class COIL100:
    def __init__(self):
        env = env_settings()
        base_path = pathlib.Path(env.synthetic_path).with_name("coil-100")

        imgs = list(base_path.glob("*.png"))
        objs = [[] for _ in range(100)]
        for img in imgs:
            obj_id = int(img.name.split('_', 1)[0][3:]) - 1
            objs[obj_id].append(str(img.resolve()))

        for val in objs:
            val.sort(key=lambda x: int(x.rsplit('_', 1)[-1][:-4]))

        processed_path = base_path / "masks"
        self.objs = self.preprocess(objs, processed_path)

    def preprocess(self, objs, dest):
        # Convert images to RGBA format where the alpha channel is 0 for the background.
        dest.mkdir(parents=True, exist_ok=True)
        new_objs = []
        for obj_imgs in objs:
            objs = []
            for obj_img in obj_imgs:
                dst = dest / pathlib.Path(obj_img).name
                objs.append(str(dst))
                if dst.exists():
                    continue 
                img = Image.open(obj_img)
                img_array = np.array(img)
                mask = utils.image_threshold_mask(img_array)
                img_array = np.append(img_array, np.atleast_3d(mask), axis=2).astype(np.uint8)  # Add alpha channel
                img = Image.fromarray(img_array)
                img.save(dst)
            new_objs.append(objs)
        return new_objs 

    def __getitem__(self, idx):
        return self.objs[idx]


class DummySequence(VOTSequence):

    def __init__(self, length=100, size=(640, 480)):
        env = env_settings()
        base = os.path.join(env.synthetic_path, "vot_dummy_%d_%d_%d" % (length, size[0], size[1]))
        if not os.path.isdir(base) or not os.path.isfile(os.path.join(base, "groundtruth.txt")):
            DummySequence._generate(base, length, size)
        super().__init__(base, None)

    @staticmethod
    def _generate(base, length, size):

        background_color = Image.fromarray(np.random.normal(15, 5, (size[1], size[0], 3)).astype(np.uint8))

        template = Image.open(os.path.join(os.path.dirname(__file__), "cow.png"))

        dir_color = os.path.join(base, "color")
        os.makedirs(dir_color, exist_ok=True)

        path_color = os.path.join(dir_color, "%08d.jpg")
        groundtruth = []

        center_x = size[0] / 2
        center_y = size[1] / 2

        radius = min(center_x - template.size[0], center_y - template.size[1])

        speed = (math.pi * 2) / length

        for i in range(length):
            frame_color = background_color.copy()

            x = int(center_x + math.cos(i * speed) * radius - template.size[0] / 2)
            y = int(center_y + math.sin(i * speed) * radius - template.size[1] / 2)

            frame_color.paste(template, (x, y), template)
            frame_color.save(path_color % (i + 1))

            groundtruth.append(Rectangle(x, y, template.size[0], template.size[1]))

        write_file(os.path.join(base, "groundtruth.txt"), groundtruth)
        metadata = {"name": "dummy", "fps" : 30, "format" : "dummy",
                          "channel.default": "color"}
        write_properties(os.path.join(base, "sequence"), metadata)


# Wrap DummySequence.
class DummyDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.sequence = DummySequence()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(self.sequence)])

    def _construct_sequence(self, sequence):
        frames = [frame.filename() for frame in sequence]
        ground_truth_rect = []
        for frame in sequence:
            gt = frame.groundtruth().convert(RegionType.RECTANGLE)
            if gt.is_empty():
                ground_truth_rect.append([-1, -1, -1, -1])
            else:
                ground_truth_rect.append([gt.x, gt.y, gt.width, gt.height])
        ground_truth_rect = np.array(ground_truth_rect)
        return Sequence(sequence.name, frames, 'Dummy', ground_truth_rect)

    def __len__(self):
        return 1


class SequentialComposite:
    def __init__(self, funcs, times=None, time_funcs=None):
        self.funcs = funcs 
        self.times = [0] + (times or [(i + 1) / len(funcs) for i in range(len(funcs))])  # Equal spacing
        self.time_funcs = time_funcs or [(lambda t: t) for _ in range(len(funcs))]

    def __call__(self, t):
        i = 1
        while i < len(self.funcs):
            if t >= self.times[i - 1] and t < self.times[i]:
                break
            i += 1
        
        t = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        t = self.time_funcs[i - 1](t)
        return self.funcs[i - 1](t)


# Map from [0, 1] to [0, 1].
class TimeFunc:
    def __call__(self, t):
        return t

class LinPeriodic:
    def __call__(self, t):
        if t <= 0.5:
            return 2 * t
        else:
            return 2 - 2 * t

class SinPeriodic:
    def __call__(self, t):
        return np.abs(np.sin(np.pi * t) ** 2)

class SmoothStep:
    def __call__(self, t):
        return 3 * t ** 2 - 2 * t ** 3


# Map from [0, 1] to [-1, 1] x [-1, 1].
class PathFunc:
    def __call__(self, t):
        return np.array([0, 0])

class CircPath:
    def __init__(self, r):
        self.r = r 

    def __call__(self, t):
        return self.r * np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]) 

class LinearPath:
    def __init__(self, direction):
        u = np.array(direction)
        self.dir = u / np.sqrt(u.dot(u))
    
    def __call__(self, t):
        return t * self.dir

class SpiralPath:
    def __call__(self, t):
        return np.array([0.8 * t * np.cos(2 * np.pi * t), 0.8 * t * np.sin(2 * np.pi * t)])

# $r = a \cos(k \theta)$
# k = p / q
class RosePath:
    def __init__(self, a=0.7, p=3, q=1):
        self.a = a 
        self.p = p 
        self.q = q 
    
    def __call__(self, t):
        period = (np.pi * self.q) if (self.p % 2 != 0) and (self.q % 2 != 0) else (2 * np.pi * self.q)
        th = t * period
        r = self.a * np.cos(self.p / self.q * th)
        return r * np.array([np.cos(th), np.sin(th)])

VPath = SequentialComposite(
    [LinearPath([np.cos(np.pi / 3), np.sin(np.pi / 3)]), LinearPath([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)])],
    time_funcs=[SinPeriodic(), SinPeriodic()])


# Map from [0, 1] to (0, inf)
class ScaleFunc:
    def __call__(self, t):
        return 1

class PeriodicScale:
    def __init__(self, range_=(0.5, 2)):
        self.range = range_
        self.f = SinPeriodic()

    def __call__(self, t):
        return self.range[0] + (self.range[1] - self.range[0]) * self.f(t)


# Map from [0, 1] to radians
class WobbleRotate:
    def __init__(self, lim=np.pi * 0.25):
        self.lim = lim  # Wobble limit in radians

    def __call__(self, t):
        return self.lim * np.sin(2 * np.pi * t)


# Map from [0, 1] to Image
class ImageFunc:
    def __init__(self, path):
        self.img = Image.open(path)

    def __call__(self, t):
        return self.img 

class ImageImage:
    def __init__(self, image):
        self.img = image 
    
    def __call__(self, t):
        return self.img

class NoiseImage:
    def __init__(self, size):
        # self.img = Image.effect_noise(size, 128)
        self.img = Image.fromarray(np.random.normal(15, 5, (size[1], size[0], 3)).astype(np.uint8))

    def __call__(self, t):
        return self.img

class ColorNoiseImage:
    def __init__(self, size, alpha=0.5):
        # Shrink and upscale for less noise
        sz = tuple((np.array(size) * alpha).astype(int))
        sigma = 128
        img = Image.fromarray(
            np.stack(
                (np.asarray(Image.effect_noise(sz, sigma)),
                np.asarray(Image.effect_noise(sz, sigma)),
                np.asarray(Image.effect_noise(sz, sigma))),
                axis=-1))
        self.img = img.resize(size, resample=Image.NEAREST)

    def __call__(self, t):
        return self.img

class ColorImage:
    def __init__(self, color, size, alpha=255):
        self.img = Image.new("RGBA", size, color=(*color, alpha))
    
    def __call__(self, t):
        return self.img 

class HSVImage:
    def __init__(self, size=(24, 24)):
        # Image.new("RGBA", (48, 48), color=(*PIL.ImageColor.getrgb('hsv(50, 100%, 100%)'), 128))
        self.img = Image.new("RGB", size, color="hsv(0, 100%, 100%)")
    
    def __call__(self, t):
        self.img.paste(f"hsv({360 * t}, 100%, 100%)", box=(0, 0, *self.img.size))
        return self.img 

# "Legacy"
class ScaleImage:
    def __init__(self, img_func, scale_func):
        self.img_func = img_func
        self.scale_func = scale_func
    
    def __call__(self, t):
        img = self.img_func(t)
        f = self.scale_func(t)
        sz = int(img.size[0] * f), int(img.size[1] * f)
        return img.resize(sz)

class AnimatedImage:
    def __init__(self, imgs):
        self.imgs = imgs  

    def __call__(self, t):
        idx = int((len(self.imgs) - 1) * t)
        return Image.open(self.imgs[idx])

class RotateImage:
    def __init__(self, img_func, rotate_func=None, **kwargs):
        self.img_func = img_func 
        self.rotate_func = rotate_func or (lambda t: 2 * np.pi * t)  # [0, 1] -> radians
        kwargs.setdefault('expand', True)
        self.kwargs = kwargs  # Image.rotate kwargs

    def __call__(self, t):
        img = self.img_func(t)
        deg = np.mod(self.rotate_func(t) * 180 / np.pi, 360)  # Allow negative values
        return img.rotate(deg, **self.kwargs)

class BlurImage:
    def __init__(self, img_func, blur_func=None):
        self.img_func = img_func
        self.blur_func = blur_func or (lambda t: 3*np.sin(np.pi*t)**2)  # [0, 1] -> blur sigma 
    
    def __call__(self, t):
        img = self.img_func(t)
        r = self.blur_func(t)
        return img.filter(filter=ImageFilter.GaussianBlur(r))


# [-1, 1] x [-1, 1] -> screen size c.s.
def world_to_pixel(x, size):
    z = 0.5 * size * (x + 1.0)
    z[1] = size[1] - z[1]  # Flip y
    return z


class SyntheticObject:
    def __init__(self, image_func, path_func=PathFunc(), time_func=TimeFunc(), scale_func=ScaleFunc(), window_size=None, window_fit=False, offset=(0, 0), **metadata):
        self.image_func = image_func 
        self.path_func = path_func
        self.time_func = time_func 
        self.scale_func = scale_func  # Scale window_size or image

        self.offset = np.array(offset)  # (center x, center y) offset in world c.s.        
        self.size = np.array(self.image_func(0).size)  # frame 0 (w, h) in screen c.s.
        
        # Object size bounding box in percent of screen size.
        # The object is fit into this window when realizing to screen c.s.
        # If None, don't fit.
        self.window_size = np.array(window_size) if window_size is not None else None 
        self.window_fit = window_fit  # If True, maintain original image aspect ratio when fitting, otherwise fill window (constant aspect ratio).

        self.meta = metadata

    def object_size(self, t, img_size, screen_size):
        # Fit and scale object
        size = np.array(img_size, dtype=np.float32)
        if self.window_size is not None:
            window = self.window_size * self.scale_func(t) * screen_size 
            if self.window_fit:
                factor = np.min(window / size)
            else:
                factor = window / size 
        else:
            factor = self.scale_func(t)
        size *= factor
        return size

    def generate(self, t, screen_size):
        t = self.time_func(t)
        img = self.image_func(t)

        sz = img.getbbox()  # Tight bbox assuming blank background
        img = img.crop(sz)
        sz = self.object_size(t, img.size, screen_size)
        img = img.resize(sz)

        return img

    def bounding_box(self, t, screen_size, tight=True):
        # Bounding box in screen coordinates.
        t = self.time_func(t)
        
        img = self.image_func(t)
        size = img.size
        if tight:
            size = img.getbbox()  # Tight bbox assuming blank background
            size = (size[2] - size[0], size[3] - size[1])
        size = self.object_size(t, size, screen_size)

        pos = world_to_pixel(self.offset + self.path_func(t), screen_size) - 0.5 * size
        pos = np.clip(pos, [0, 0], screen_size)  # Clamp position
        size = np.minimum(size + pos, screen_size) - pos  # Clamp size 
        return [*pos, *size]


class SyntheticSequence:
    def __init__(self, obj, bg, length, name, screen_size=None, save_path=None):
        self.obj = obj 
        self.bg = bg
        self.length = length

        # Resize background
        self.bg.window_fit = False
        if screen_size is not None:
            self.bg.window_size = np.array([1, 1])
        self.screen_size = screen_size or self.bg.size
        self.screen_size = np.array(self.screen_size)

        self.name = f"{name}_{length}_{self.screen_size[0]}_{self.screen_size[1]}"
        if save_path is None:
            save_path = pathlib.Path(env_settings().synthetic_path) 
        self.save_path = save_path / self.name

    def generate_frame(self, idx, write=True, force=False, debug=False):
        t = idx / self.length
        bbox = self.obj.bounding_box(t, self.screen_size)

        path = self.save_path / f"{idx:06}.jpg"
        if not force and path.exists():
            return path, bbox, Image.open(path)

        bg_img = self.bg.generate(t, self.screen_size)
        obj_img = self.obj.generate(t, self.screen_size)

        img = bg_img.copy()  # Cool effect without copy
        img.paste(obj_img, (int(bbox[0]), int(bbox[1])), obj_img)

        if debug:
            draw = ImageDraw.Draw(img)
            center = [bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]]
            draw.rectangle([center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2], fill='green')
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline='red', width=2)

        if write:
            img.save(path)
        else:
            path = None
        return path, bbox, img

    def generate(self, **kwargs):
        self.save_path.mkdir(parents=True, exist_ok=True)
        frames = [self.generate_frame(i, write=True, **kwargs)[:2] for i in range(self.length)]
        return [str(f[0]) for f in frames], [f[1] for f in frames]

    def ground_truth_rect(self, idx):
        t = idx / self.length 
        return self.obj.bounding_box(t, self.screen_size)
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.name 


class SyntheticDataset(BaseDataset):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def get_sequence_list(self, **kwargs):
        return SequenceList([SyntheticDataset._construct_sequence(s, **kwargs) for s in self.sequences])

    @staticmethod
    def _construct_sequence(sequence, **kwargs):
        frames, gt = sequence.generate(**kwargs)
        ground_truth_rect = np.array(gt)
        return Sequence(sequence.name, frames, "Synthetic", ground_truth_rect)

    def clear(self):
        for s in self.sequences:
            shutil.rmtree(s.save_path)

    def __len__(self):
        return len(self.sequences)

    def __repr__(self):
        return f"<SyntheticDataset: {repr(self.sequences)}>"


# Merge lists of same shape.
def shape_merge_list(xs):
    out = []
    for es in zip(*xs):
        if isinstance(es[0], list):
            out.append(shape_merge_list(list(es)))
        else:
            out.extend(es)
    return out

def shape_merge_dict(xs):
    out = {}
    for key, value in xs[0].items():
        if isinstance(value, dict):
            out[key] = shape_merge_dict([x[key] for x in xs])
        else:
            out[key] = [x[key] for x in xs]
    return out

def attach_metadata(obj, **meta):
    for key, value in meta.items():
        setattr(obj, key, value)
    return obj

def action_curriculum(obj_img):
    return {
        'static': SyntheticObject(
            obj_img,
            window_size=(0.75, 0.75),
            window_fit=True,
            desired_length=10),
        'horizontal': SyntheticObject( 
            obj_img, 
            path_func=LinearPath([1, 0]), 
            time_func=SinPeriodic(), 
            offset=(-0.5, 0),
            window_size=(0.4, 0.4),
            window_fit=True,
            desired_length=30),
        'vertical': SyntheticObject(
            obj_img, 
            path_func=LinearPath([0, 1]), 
            time_func=SinPeriodic(), 
            offset=(0, -0.5),
            window_size=(0.4, 0.4),
            window_fit=True,
            desired_length=30),
        'diagonal': SyntheticObject(
            obj_img,
            path_func=VPath,
            offset=(0, -0.5),
            window_size=(0.4, 0.4),
            window_fit=True,
            desired_length=50),
        'expansion': SyntheticObject(
            obj_img,
            scale_func=PeriodicScale(range_=(0.25, 1.25)),
            window_size=(0.5, 0.5),
            window_fit=True,
            desired_length=40),
        'circ': SyntheticObject(
            obj_img,
            path_func=CircPath(0.5),
            time_func=SinPeriodic(),
            scale_func=PeriodicScale(range_=(0.5, 1.25)),
            window_size=(0.35, 0.35),
            window_fit=True,
            desired_length=80),
        'spiral': SyntheticObject(
            obj_img,
            path_func=SpiralPath(),
            time_func=SinPeriodic(),
            scale_func=PeriodicScale(range_=(0.25, 1.25)),
            window_size=(0.35, 0.35),
            window_fit=True,
            desired_length=100),
        'rose3': SyntheticObject(
            obj_img,
            path_func=RosePath(a=0.5, p=3, q=1),
            time_func=SmoothStep(),
            scale_func=PeriodicScale(range_=(0.25, 1.25)),
            window_size=(0.35, 0.35),
            window_fit=True,
            desired_length=120),
        'rose3_5': SyntheticObject(
            obj_img,
            path_func=RosePath(a=0.5, p=3, q=5),
            time_func=SmoothStep(),
            scale_func=PeriodicScale(range_=(0.3, 1.2)),
            window_size=(0.4, 0.4),
            window_fit=True,
            desired_length=144)
    }

def background_curriculum():
    size = (107 * 3, 107 * 3)
    vot = VOTDatasetWrapped()

    return {
        'black': SyntheticObject(
            ImageImage(Image.new("RGB", size, 'black')),
            desired_length=10),
        'noise': SyntheticObject(
            NoiseImage(size),
            desired_length=10), 
        'hsv': SyntheticObject(
            HSVImage(size),
            desired_length=40),
        'colornoise': SyntheticObject(
            ColorNoiseImage(size),
            desired_length=20),
        'fish1': SyntheticObject(
            AnimatedImage(vot['fish1'].frames),
            desired_length=len(vot['fish1'].frames) // 4),
        'fish2': SyntheticObject(
            AnimatedImage(vot['fish2'].frames),
            desired_length=len(vot['fish2'].frames) // 4),
        'butterfly': SyntheticObject(
            AnimatedImage(vot['butterfly'].frames),
            desired_length=len(vot['butterfly'].frames) // 4),
        'rabbit': SyntheticObject(
            AnimatedImage(vot['rabbit'].frames),
            desired_length=len(vot['rabbit'].frames) // 4),
        'motocross1': SyntheticObject(
            AnimatedImage(vot['motocross1'].frames),
            desired_length=len(vot['motocross1'].frames) // 4)                                    
    }

def object_curriculum(filtered=None):
    def make_obj(obj):
        return {
            'anim': AnimatedImage(obj),
            'wobble': 
                RotateImage(
                    AnimatedImage(obj), 
                    rotate_func=WobbleRotate(),
                    expand=True),
            'blurwobble': 
                RotateImage(
                    BlurImage(
                        AnimatedImage(obj)),
                    rotate_func=WobbleRotate(),
                    expand=True),
            'full': 
                RotateImage(
                    BlurImage(
                        AnimatedImage(obj)),
                    expand=True),
            'blurwobble2': 
                RotateImage(
                    BlurImage(
                        AnimatedImage(obj)),
                    rotate_func=WobbleRotate(np.pi / 2),
                    expand=True)
        }
    
    my_objs = MyObjects()
    coil = COIL100()
    # objs = [*my_objs.objs.values(), coil[73]]
    objs = {
        **my_objs.objs,
        'duck': coil[73],
    }
    if filtered is not None:
        objs = { k: v for k, v in objs.items() if k in filtered }
    # return { obj: make_obj(frames) for obj, frames in objs.items() }, { obj: len(frames) // 2 for obj, frames in objs.items() }
    return shape_merge_dict([make_obj(obj) for obj in objs.values()]), [(obj, len(frames) // 2) for obj, frames in objs.items()]

def make_curriculum(object_filter=None):
    bg_cs = background_curriculum()
    obj_cs, obj_infos = object_curriculum(object_filter)
    # action_cs = [action_curriculum(obj) for obj_c in obj_cs for obj in obj_c]
    
    save_path = pathlib.Path(env_settings().synthetic_path)
    curriculum = []

    # Curriculum goal 1: recognize objects with simple movements on simple backgrounds
    seqs = []
    curriculum_1 = [('noise', 'anim', a) for a in ('static', 'vertical', 'horizontal', 'expansion', 'diagonal')]
    for bg_name, obj_name, action_name in curriculum_1:
        bg = bg_cs[bg_name]
        obj_imgs = obj_cs[obj_name]
        for obj_idx, obj_img in enumerate(obj_imgs):
            actor = action_curriculum(obj_img)[action_name]

            length = max(bg.meta.get('desired_length', 10), actor.meta.get('desired_length', 10), obj_infos[obj_idx][1])
            # print(bg_name, obj_name, obj_infos[obj_idx], action_name, length)
            seq = SyntheticSequence(
                actor, 
                bg,
                length,
                f"{bg_name}_{obj_name}_{obj_infos[obj_idx][0]}_{action_name}",
                save_path=save_path / "curriculum")
            seqs.append(seq)
    curriculum.append(seqs)

    # Curriculum goal 2: harder movement on colored backgrounds
    seqs = []
    curriculum_2 = [('hsv', 'wobble', 'circ'), ('colornoise', 'blurwobble', 'rose3')]
    for bg_name, obj_name, action_name in curriculum_2:            
        bg = bg_cs[bg_name]
        obj_imgs = obj_cs[obj_name]
        for obj_idx, obj_img in enumerate(obj_imgs):
            actor = action_curriculum(obj_img)[action_name]

            length = max(bg.meta.get('desired_length', 10), actor.meta.get('desired_length', 10), obj_infos[obj_idx][1])
            # print(bg_name, obj_name, obj_infos[obj_idx], action_name, length)
            seq = SyntheticSequence(
                actor, 
                bg,
                length,
                f"{bg_name}_{obj_name}_{obj_infos[obj_idx][0]}_{action_name}",
                save_path=save_path / "curriculum")
            seqs.append(seq)
    curriculum.append(seqs)

    # Curriculum goal 3: hard
    # seqs = []
    # for bg_name, bg in bg_cs['video'].items():
    #     for obj_name, obj_imgs in obj_cs['difficult'].items():
    #         for obj_idx, obj_img in enumerate(obj_imgs):
    #             action_cs = action_curriculum(obj_img)
    #             for action_name, actor in action_cs['difficult'].items():
    #                 length = max(bg.meta.get('desired_length', 10), actor.meta.get('desired_length', 10), obj_infos[obj_idx][1])
    #                 # print(bg_name, obj_name, obj_infos[obj_idx], action_name, length)
    #                 seq = SyntheticSequence(
    #                     actor, 
    #                     bg,
    #                     length,
    #                     f"{bg_name}_{obj_name}_{obj_infos[obj_idx][0]}_{action_name}",
    #                     save_path=save_path / "curriculum")
    #                 seqs.append(seq)
    # curriculum.append(seqs)

    # Target sequences: how fast can target sequences be mastered with/without a curriculum?
    # Target sequence backgrounds and/or objects should be different from training set examples....
    # TODO also try setting the last curriculum as the target ...
    target_seqs = []
    target = [('fish1', 'blurwobble2', 'rose3_5'), ('fish2', 'blurwobble2', 'rose3_5')]
    for bg_name, obj_name, action_name in target:
        bg = bg_cs[bg_name]
        obj_imgs = obj_cs[obj_name]
        for obj_idx, obj_img in enumerate(obj_imgs):
            actor = action_curriculum(obj_img)[action_name]

            length = max(bg.meta.get('desired_length', 10), actor.meta.get('desired_length', 10), obj_infos[obj_idx][1])
            # print(bg_name, obj_name, obj_infos[obj_idx], action_name, length)
            seq = SyntheticSequence(
                actor, 
                bg,
                length,
                f"{bg_name}_{obj_name}_{obj_infos[obj_idx][0]}_{action_name}",
                save_path=save_path / "curriculum")
            target_seqs.append(seq)

    # test_seqs = [vot['fish1'], vot['fish2']]
    evaluation_seqs = []
    evaluation = [('motocross1', 'blurwobble2', 'rose3_5'), ('rabbit', 'wobble', 'rose3')]
    for bg_name, obj_name, action_name in evaluation:
        bg = bg_cs[bg_name]
        obj_imgs = obj_cs[obj_name]
        for obj_idx, obj_img in enumerate(obj_imgs):
            actor = action_curriculum(obj_img)[action_name]

            length = max(bg.meta.get('desired_length', 10), actor.meta.get('desired_length', 10), obj_infos[obj_idx][1])
            # print(bg_name, obj_name, obj_infos[obj_idx], action_name, length)
            seq = SyntheticSequence(
                actor, 
                bg,
                length,
                f"{bg_name}_{obj_name}_{obj_infos[obj_idx][0]}_{action_name}",
                save_path=save_path / "curriculum")
            evaluation_seqs.append(seq)

    return [SyntheticDataset(c) for c in curriculum], SyntheticDataset(target_seqs), SyntheticDataset(evaluation_seqs)

def _old_sequences():
    # Sorted by 'difficulty'
    SEQUENCES = [
        SyntheticSequence(
            SyntheticObject(ColorImage((255, 0, 0), (50, 50))),
            SyntheticObject(NoiseImage((480, 480))),
            20,
            "static"),
        SyntheticSequence(
            SyntheticObject(ColorImage((0, 255, 0), (60, 60)), LinearPath([1, 0]), SinPeriodic(), offset=(-0.5, 0)),
            SyntheticObject(NoiseImage((480, 480))),
            50,
            "horizontal"),
        SyntheticSequence(
            SyntheticObject(ColorImage((0, 255, 0), (60, 60)), LinearPath([0, 1]), SinPeriodic(), offset=(0, -0.5)),
            SyntheticObject(NoiseImage((480, 480))),
            50,
            "vertical"),
        SyntheticSequence(
            SyntheticObject(ImageFunc(os.path.join(os.path.dirname(__file__), "cow.png")), CircPath(0.5), SinPeriodic()),
            SyntheticObject(NoiseImage((480, 480))),
            100,
            "circ"
        ),
        SyntheticSequence(
            SyntheticObject(
                ScaleImage(
                    ImageFunc(os.path.join(os.path.dirname(__file__), "cow.png")),
                    PeriodicScale(range_=(0.5, 1.5))),
                CircPath(0.5), 
                SinPeriodic()),
            SyntheticObject(NoiseImage((480, 480))),
            100,
            "circ_scale"
        )
    ]
    return SEQUENCES
