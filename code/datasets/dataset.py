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

sys.path.insert(0, '/home/nirmal/sm/code')

from datasets.random_erasing import RandomErasing
from datasets.video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_resized_crop_with_shift, horizontal_flip,   
)
from datasets.volume_transforms import ClipToTensor
from utils import ZipReader

class Phoenix2014Video(Dataset):
    def __init__(
            self, anno_path: str='', gloss_to_id_path: str='', video_path: str='', mode: str='train', clip_len: int=256,
            aug_size: tuple=(210, 210), target_size: tuple=(224, 224), gloss_tokenizer=None, args=None
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

        self.gloss_tokenizer = gloss_tokenizer
        
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
            # for i in range(anno['num_frames']):
            #     self.img_paths.append(f"{self.video_path}@{anno['name']}.avi_pid0_fn{i:06d}-0.png") # 'data/phoenix-2014/phoenix-2014-videos.zip@fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png', ...
            frames = [
                f"{self.video_path}@{anno['name']}.avi_pid0_fn{i:06d}-0.png" 
                for i in range(anno['num_frames'])
            ]
            self.img_paths.append(frames)
                
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
        
        # Set tmin and tmax based on mode
        if self.mode == 'train':
            self.tmin = self.args.t_min if self.args and hasattr(self.args, 't_min') else 1
            self.tmax = self.args.t_max if self.args and hasattr(self.args, 't_max') else 1
        else:
            self.tmin = 1
            self.tmax = 1

    def __getitem__(self, idx):
        args = self.args
        video_name = self.videos[idx]
        gloss_seq = self.glosses[idx]
        n_frames = self.n_frames[idx]

        all_frame_paths = self.img_paths[idx]

        selected_frame_paths = self._sample_frames(all_frame_paths, n_frames)

        # print(f"Selected frames count: {selected_frame_paths}")

        # Read images and keep their paths
        frames = [(path, self.read_img(path)) for path in selected_frame_paths]
        frames.sort(key=lambda x: x[0])
        # Sort frames by path to ensure the correct order
        frames = [frame[1] for frame in frames]  # Extract images only

        # Tokenize gloss sequence
        tokenized_gloss = self.gloss_tokenizer([gloss_seq])
        gloss_ids = tokenized_gloss['gloss_labels'][0]
        gloss_len = len(gloss_ids)
        
        # Apply transformations
        if self.mode == 'train':
            frames = self._aug_frame(frames, args)
            n_frames = frames.shape[1]
            return frames, n_frames, gloss_ids, gloss_len, idx
        
        elif self.mode == 'validation':
            frames = self.data_transform(frames)
            n_frames = frames.shape[1]
            return frames, n_frames, gloss_ids, gloss_len, idx
        
        elif self.mode == 'test':
            frames = self.data_transform(self.data_resize(frames))
            n_frames = frames.shape[1]
            # print(f"[DEBUG] From Datasets: frames, n_frames, gloss_ids, gloss_len, idx: {frames.shape, n_frames, gloss_ids, gloss_len, idx}")
            return frames, n_frames, gloss_ids, gloss_len, idx
    
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

        # for i, frame in enumerate(buffer.permute(1, 0, 2, 3)):
        #     transforms.ToPILImage()(frame).save(f"augmented_frame_{i}.png")

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
        
    #    # 역정규화하여 디버깅용 이미지 저장
    #     for i, frame in enumerate(buffer.permute(1, 0, 2, 3)):  # (C, T, H, W) -> (T, C, H, W)
    #         denormalized_frame = tensor_denormalize(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         transforms.ToPILImage()(denormalized_frame).save(f"debug_denorm_frame_{i}.png")

    #         if i == 10:  # 첫 10개만 저장
    #             break

        
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
        frame_paths = sorted(frame_paths)
        # 원하는 tmin, tmax, max_num_frames 설정
        # tmin = self.args.t_min if self.args and hasattr(self.args, 't_min') else 1
        # tmax = self.args.t_max if self.args and hasattr(self.args, 't_max') else 1

        frame_index = get_selected_indexs(
            vlen=n_frames, 
            tmin=self.tmin, 
            tmax=self.tmax, 
            max_num_frames=self.clip_len
        )
        selected_frame_paths = [frame_paths[i] for i in frame_index]

        return selected_frame_paths
    
    # def _sample_frames(self, frame_paths, n_frames):
    #     """
    #     _sample_frames를 clip_len 고정 방식으로 수정:
    #     - n_frames >= self.clip_len이면 균등 간격 샘플링
    #     - n_frames < self.clip_len이면 마지막 프레임을 반복해 padding
    #     """
    #     clip_len = self.clip_len
    #     frame_paths = sorted(frame_paths)

    #     if n_frames == 0:
    #         # 에지 케이스(프레임이 없는 경우) - 여기서는 일단 빈 리스트 반환 또는 오류 처리
    #         return []

    #     if n_frames >= clip_len:
    #         # 균등 간격으로 clip_len개 프레임 선택
    #         # 예: n_frames=300, clip_len=100이면, step=3 -> [0, 3, 6, ..., 297]
    #         step = n_frames / clip_len
    #         frame_index = [int(round(i * step)) for i in range(clip_len)]
    #         # 혹은 int(math.floor(i * step)) 등을 취할 수도 있음
    #         # 인덱스가 n_frames-1을 넘지 않도록 보정
    #         frame_index = [min(idx, n_frames - 1) for idx in frame_index]

    #     else:
    #         # 프레임 수가 부족하므로, 부족분을 마지막 프레임으로 반복
    #         frame_index = list(range(n_frames))
    #         while len(frame_index) < clip_len:
    #             frame_index.append(n_frames - 1)  # 마지막 프레임 복제

    #     selected_frame_paths = [frame_paths[i] for i in frame_index]
    #     return selected_frame_paths

    def __len__(self):
        return len(self.annotations)
    
# def get_selected_indexs(vlen, tmin=1, tmax=1, max_num_frames=400):
#     """
#     vlen: frame 수
#     """
#     if tmin==1 and tmax==1:
#         if vlen <= max_num_frames:
#             frame_index = np.arange(vlen)
#             valid_len = vlen
#         else:
#             sequence = np.arange(vlen)
#             an = (vlen - max_num_frames)//2
#             en = vlen - max_num_frames - an
#             frame_index = sequence[an: -en]
#             valid_len = max_num_frames

#         if (valid_len % 2) != 0:
#             valid_len -= (valid_len % 2)
#             frame_index = frame_index[:valid_len]

#         assert len(frame_index) == valid_len, (frame_index, valid_len)
#         return frame_index
    
#     min_len = int(tmin*vlen)
#     max_len = min(max_num_frames, int(tmax*vlen))
#     selected_len = np.random.randint(min_len, max_len+1)
    
#     if (selected_len % 2) != 0:
#         selected_len += (2-(selected_len % 2))
#     if selected_len<=vlen: 
#         selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
#     else: 
#         copied_index = np.random.randint(0,vlen,selected_len-vlen)
#         selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

#     if selected_len <= max_num_frames:
#         frame_index = selected_index
#         valid_len = selected_len
#     else:
#         assert False, (vlen, selected_len, min_len, max_len)
#     assert len(frame_index) == valid_len, (frame_index, valid_len)
#     return frame_index

def get_selected_indexs(vlen, tmin=1, tmax=1, max_num_frames=400):
    """
    temporal augmentation(tmin, tmax)에 따라 선택된 프레임 인덱스를
    최종적으로 max_num_frames로 맞추어 반환합니다.
    
    vlen: 전체 프레임 수
    tmin, tmax: temporal augment 범위
    max_num_frames: 최종적으로 고정할 프레임 수
    """
    # tmin==1, tmax==1인 경우는 별도 로직(기존과 동일) --------------------------------
    if tmin == 1 and tmax == 1:
        if vlen <= max_num_frames:
            # 모든 프레임을 선택
            frame_index = np.arange(vlen)
            
            if vlen < max_num_frames:
                # 부족한 프레임 수만큼 마지막 프레임을 반복하여 패딩
                pad_length = max_num_frames - vlen
                pad_indices = np.full(pad_length, vlen - 1, dtype=int)
                frame_index = np.concatenate([frame_index, pad_indices])
                
            # 최종적으로 frame_index의 길이가 max_num_frames인지 확인
            assert len(frame_index) == max_num_frames, f"frame_index 길이가 {max_num_frames}이 아닙니다. 실제 길이: {len(frame_index)}"
        
        else:
            # vlen이 max_num_frames보다 클 때는 중앙에서 max_num_frames만큼 선택
            sequence = np.arange(vlen)
            an = (vlen - max_num_frames) // 2
            en = vlen - max_num_frames - an
            frame_index = sequence[an: -en]
            
        return frame_index
    
    # if tmin and tamx != 1: random temporal augment logic -----------------------
    min_len = int(tmin * vlen)
    max_len = min(max_num_frames, int(tmax * vlen))

    # print(f"======================================Min_len: {min_len}, Max_len: {max_len}")

    # select a random length from range min_len to max_len
    chosen_len = np.random.randint(min_len, max_len + 1)

    # chosen_len이 vlen 이하이면 vlen 중에서 chosen_len개를 무작위로 뽑음
    if chosen_len <= vlen:
        selected_index = sorted(np.random.permutation(np.arange(vlen))[:chosen_len])
    else:
        # chosen_len이 vlen보다 크면, 부족한 만큼 중복 샘플링해서 붙인다
        extra = chosen_len - vlen
        copied_index = np.random.randint(0, vlen, extra)
        selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

    # 이제 selected_index의 길이가 chosen_len이지만, 최종적으로 max_num_frames로 맞춤
    if chosen_len < max_num_frames:
        # chosen_len이 부족하면 마지막 프레임 반복
        while len(selected_index) < max_num_frames:
            selected_index.append(selected_index[-1])
        frame_index = np.array(selected_index)
    
    elif chosen_len > max_num_frames:
        # chosen_len이 너무 많으면 균등 간격으로 서브샘플링
        step = chosen_len / max_num_frames
        new_index = [int(round(i * step)) for i in range(max_num_frames)]
        new_index = [min(idx, chosen_len - 1) for idx in new_index]
        frame_index = np.array([selected_index[i] for i in new_index])
    
    else:
        # chosen_len == max_num_frames인 경우
        frame_index = np.array(selected_index)

    # 최종적으로 frame_index의 길이가 max_num_frames인지 확인
    assert len(frame_index) == max_num_frames, f"frame_index 길이가 {max_num_frames}이 아닙니다. 실제 길이: {len(frame_index)}"
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

def tensor_denormalize(tensor: torch.Tensor, mean, std):
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(-1, 1, 1)  # (C, 1, 1)
    if isinstance(std, list):
        std = torch.tensor(std).view(-1, 1, 1)  # (C, 1, 1)
    
    tensor = tensor * std + mean  # 역정규화
    return tensor


def build_dataset(modal, gloss_tokenizer, is_train, is_test, args):
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
                clip_len=args.num_frames, aug_size=args.aug_size, target_size=args.video_size, gloss_tokenizer=gloss_tokenizer, args=args
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


if __name__ == '__main__':
    from run import get_args
    from tokenizer import GlossTokenizer_S2G
    import pickle

    args, _ = get_args()

    print(f"Args: {args}")
    print(f"Args: {args.gloss_to_id_path}")

    with open(args.gloss_to_id_path, 'rb') as f:
        gloss_to_id = pickle.load(f)

    tokenizer_cfg = {
        'gloss2id_file': args.gloss_to_id_path,
    }

    gloss_tokenizer = GlossTokenizer_S2G(tokenizer_cfg)

    mode = 'validation'

    # Instantiate the dataset for different modes
    dataset_train = Phoenix2014Video(
        anno_path=f"{args.metadata_path}/phoenix-2014.{mode}",
        gloss_to_id_path=args.gloss_to_id_path,
        video_path=args.video_path,
        mode= mode,
        clip_len=args.num_frames,
        target_size=args.video_size,
        aug_size=args.aug_size,
        gloss_tokenizer=gloss_tokenizer,
        args=args
    )

    # Function to inspect dataset outputs
    def inspect_dataset(dataset, mode):
        print(f"\nInspecting {mode} dataset:")

        for idx in range(5):
            sample = dataset[idx]

            if mode == "train":
                frames, num_frames, gloss_ids, gloss_len, video_name = sample
                print(f"Sample {idx + 1}:")
                print(f"Frames shape: {frames.shape}")
                print(f"Number of frames: {num_frames}")
                print(f"Gloss IDs: {gloss_ids}")
                print(f"Gloss length: {gloss_len}")
                print(f"Video name: {video_name}")
            elif mode in ["validation", "test"]:
                frames, num_frames, gloss_ids, gloss_len, idx = sample
                print(f"Frames shape: {frames.shape}")
                print(f"Number of frames: {num_frames}")
                print(f"Gloss IDs: {gloss_ids}")
                print(f"Gloss length: {gloss_len}")
                print(f"Index: {idx}")

            # Inspect each dataset
    inspect_dataset(dataset_train, mode=mode)