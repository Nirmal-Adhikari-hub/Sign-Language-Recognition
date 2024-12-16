from typing import Optional
import os, io
import sys
import warnings
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import zipfile
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_resized_crop_with_shift, horizontal_flip,   
)
from .volume_transforms import ClipToTensor
from .tokenizer import GlossTokenizer_S2G
from utils import ZipReader

class Phoenix2014Video(Dataset):
    def __init__(
            self, anno_path: str='', gloss_to_id_path: str='', video_path: str='', mode: str='train', clip_len: int=256,
            aug_size: tuple=(210, 210), target_size: tuple=(224, 224), args=None
        ):
        """
        Please Refer to TwostreamNetwork/origin/dataset

        Choose `clip_len` of images in each folder
            if clip_len <= max_clip_len
                then pad
            elif clip_len > max_clip_len:
                then choose last ones
        """
        self.anno_path = anno_path # Annotations path
        self.video_path = video_path # Videos path
        self.mode = mode # train, dev, test
        self.clip_len = clip_len # Length of the clip for model's input
        self.aug_size = aug_size # Preprocessed size
        self.target_size = target_size # Input size
        self.args = args
        self.aug = False


        if self.mode == 'train':
            self.aug = True
            if self.args is not None and hasattr(self.args, "reprob") and self.args.reprob > 0.:
                self.rand_erase = True
            else:
                self.rand_erase = False

        tokenizer_cfg = {
            'gloss2id_file': gloss_to_id_path,
        }
        self.gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg)
        
        import gzip, pickle

        # Load gloss-to-id lookup table
        with open(gloss_to_id_path, 'rb') as f:
            self.gloss_to_id = pickle.load(f)

        with gzip.open(anno_path) as f:
            self.annotations = pickle.load(f)

        self.videos, self.img_paths, self.glosses, self.n_frames = [], [], [], []
        for anno in self.annotations:
            self.videos.append(anno['name']) # 'fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute', ...
            self.glosses.append(anno['gloss']) # '__ON__ LIEB ZUSCHAUER ABEND WINTER GESTERN loc-NORD SCHOTTLAND loc-REGION UEBERSCHWEMMUNG AMERIKA IX', ...
            self.n_frames.append(anno['num_frames']) # 176, ...
            for i in range(anno['num_frames']):
                self.img_paths.append(f"{self.video_path}@{anno['name']}.avi_pid0_fn{i:06d}-0.png") # 'data/phoenix-2014/phoenix-2014-videos.zip@fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png', ...

        if mode == 'train':
            pass
        elif mode == 'validation':
            self.data_transform = Compose([
                CenterCrop(size=aug_size),
                Resize(self.target_size, interpolation='bilinear'),
                ClipToTensor(),
                Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]) # ImageNet
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(self.target_size, interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]) # ImageNet
            ])

    def __getitem__(self, idx):
        args = self.args
        video_name = self.videos[idx]
        gloss_seq = self.glosses[idx]
        n_frames = self.n_frames[idx]

        all_frame_paths = [
            path for path in self.img_paths if video_name in path
        ]
        
        selected_frame_paths = self._sample_frames(all_frame_paths, n_frames)

        frames = [self.read_img(path) for path in selected_frame_paths] # ndarray List:[1(H,W,C), 2(H,W,C), ...T(H,W,C)]
        
        tokenized_gloss = self.gloss_tokenizer([gloss_seq])
        gloss_ids = tokenized_gloss['gloss_labels'][0]
        
        # Apply transformations
        if self.mode == 'train':
            frames = self._aug_frame(frames, args)
            return frames, gloss_ids, idx
        
        elif self.mode == 'validation':
            frames = self.data_transform(frames)
            return frames, gloss_ids, idx
        
        elif self.mode == 'test':
            frames = self.data_transform(self.data_resize(frames))
            return frames, gloss_ids, idx
    
    def _aug_frame(self, buffer, args):
        """
        Perform random augmentations on the video frames.
        buffer: List of frames (H, W, C) as NumPy arrays.
        Returns: Augmented tensor of shape (C, T, H, W).
        """
        # Random augmentation transform
        aug_transform = create_random_augment(
            input_size=self.aug_size,
            auto_augment=self.args.aa if self.args and hasattr(self.args, "aa") else None,
            interpolation=self.args.train_interpolation if self.args and hasattr(self.args, "train_interpolation") else "bilinear",
        )
        
        # Convert NumPy frames to PIL images for transformation
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]  # (T, H, W, C)
        buffer = aug_transform(buffer)  # Apply augmentation (returns list of PIL Images)
        
        # Convert back to tensors
        buffer = [transforms.ToTensor()(img) for img in buffer]  # (T, C, H, W)
        buffer = torch.stack(buffer)  # (T, C, H, W)
        
        buffer = buffer.permute(0, 2, 3, 1) # (T, H, W, C)
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # spatial sampling을 위해 shape 변형
        buffer = buffer.permute(3, 0, 1, 2) # (C, T, H, W)

        scl, asp = (
            [0.75, 1],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            frames=buffer, target_height=self.target_size[0], target_width=self.target_size[1],
            random_horizontal_flip=True, scale=scl, aspect_ratio=asp
        )
        
        if self.rand_erase:
            random_erasing = RandomErasing(
                args.reprob, mode=args.remode, max_count=args.recount,
                num_splits=args.recount, device='cpu'
            )
            # random erasing을 위해 shape 변형
            buffer = buffer.permute(1, 0, 2, 3) # (T, C, H, W)
            buffer = random_erasing(buffer)
            # 모델의 입력값으로 사용하기 위해 shape 변형
            buffer = buffer.permute(1, 0, 2, 3) # (C, T, H, W)
        
        return buffer



    def read_img(self, path):
        """
        Use ZipReader to read an image from a zip-style path.
        """
        try:
            img_data = ZipReader.read(path)
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            return np.array(image)  # Convert to NumPy array
        except Exception as e:
            raise RuntimeError(f"Failed to read image {path}: {e}")

        

    def _sample_frames(self, frame_paths, n_frames):
        # 원하는 tmin, tmax, max_num_frames 설정
        tmin = self.args.t_min if self.args and hasattr(self.args, 't_min') else 1
        tmax = self.args.t_max if self.args and hasattr(self.args, 't_max') else 1

        frame_index = get_selected_indexs(
            vlen=n_frames, 
            tmin=tmin, 
            tmax=tmax, 
            max_num_frames=self.clip_len
        )
        selected_frame_paths = [frame_paths[i] for i in frame_index]
        return selected_frame_paths
    
    def __len__(self):
        return len(self.annotations)

def get_selected_indexs(vlen, tmin=1, tmax=1, max_num_frames=400):
    """
    vlen: frame 수
    """
    if tmin==1 and tmax==1:
        if vlen <= max_num_frames:
            frame_index = np.arange(vlen)
            valid_len = vlen
        else:
            sequence = np.arange(vlen)
            an = (vlen - max_num_frames)//2
            en = vlen - max_num_frames - an
            frame_index = sequence[an: -en]
            valid_len = max_num_frames

        if (valid_len % 2) != 0:
            valid_len -= (valid_len % 2)
            frame_index = frame_index[:valid_len]

        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index
    
    min_len = int(tmin*vlen)
    max_len = min(max_num_frames, int(tmax*vlen))
    selected_len = np.random.randint(min_len, max_len+1)
    
    if (selected_len % 2) != 0:
        selected_len += (2-(selected_len % 2))
    if selected_len<=vlen: 
        selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
    else: 
        copied_index = np.random.randint(0,vlen,selected_len-vlen)
        selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

    if selected_len <= max_num_frames:
        frame_index = selected_index
        valid_len = selected_len
    else:
        assert False, (vlen, selected_len, min_len, max_len)
    assert len(frame_index) == valid_len, (frame_index, valid_len)
    return frame_index

def spatial_sampling(
        frames: torch.Tensor, target_height: int=224, target_width: int=224,
        random_horizontal_flip: bool=True, scale: Optional[list]=None, aspect_ratio: Optional[list]=None
    ):
    """
    Random resized crop -> Horizontal flip
    """
    transform_func = random_resized_crop_with_shift
    frames = transform_func(
        images=frames, target_height=target_height, target_width=target_width, scale=scale, ratio=aspect_ratio
    )
    if random_horizontal_flip:
        frames, _ = horizontal_flip(.5, frames)
    
    return frames

def tensor_normalize(tensor: torch.Tensor, mean, std):
    tensor = tensor.float()
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def build_dataset(modal, is_train, is_test, args):
    assert modal in ('video', 'keypoint')
    print(f"Use Dataset: {args.dataset}")

    if args.dataset == 'phoenix-2014':
        mode = None
        anno_path = None

        if modal == 'video':
            if is_train == True:
                mode = 'train'
                anno_path = os.path.join(args.metadata_path, 'phoenix-2014.train')
            elif is_test == True:
                mode = 'test'
                anno_path = os.path.join(args.metadata_path, 'phoenix-2014.test')
            else:
                mode = 'validation'
                anno_path = os.path.join(args.metadata_path, 'phoenix-2014.dev')
        
            dataset = Phoenix2014Video(
                anno_path=anno_path, gloss_to_id_path=args.gloss_to_id_path, video_path=args.video_path, mode=mode,
                clip_len=args.n_frames, aug_size=args.aug_size, target_size=args.video_size, args=args
            )
        elif modal == 'keypoint':
            if is_train == True:
                mode = 'train'
                anno_path = os.path.join(args.metadata_path, 'mm_train.csv')
            elif is_test == True:
                mode = 'test'
                anno_path = os.path.join(args.metadata_path, 'mm_test.csv')
            else:
                mode = 'validation'
                anno_path = os.path.join(args.metadata_path, 'mm_val.csv')
        
            dataset = Phoenix2014Keypoint(
                ...
            )
    else:
        print(f"Wrong: {args.dataset}")
        raise NotImplementedError
    
    return dataset