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
# from .random_erasing import RandomErasing
sys.path.append('/home/nirmal/sm/code')
# print(sys.path)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_resized_crop_with_shift, horizontal_flip,   
)
from volume_transforms import ClipToTensor
from utils import ZipReader

class Phoenix2014Video(Dataset):
    def __init__(
            self, anno_path: str='', gloss_to_id_path: str='', video_path: str='', mode: str='train', clip_len: int=256,
            aug_size: tuple=(256, 256), target_size: tuple=(224, 224), args=None
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

        
        import gzip, pickle

        # Load gloss-to-id lookup table
        with open(gloss_to_id_path, 'rb') as f:
            self.gloss_to_id = pickle.load(f)

        with gzip.open(anno_path) as f:
            annotations = pickle.load(f)

        self.videos, self.img_paths, self.glosses, self.n_frames = [], [], [], []
        for anno in annotations:
            self.videos.append(anno['name']) # 'fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute', ...
            gloss_ids = [self.gloss_to_id[word] for word in anno['gloss'].split()] # '__ON__ LIEB ZUSCHAUER ABEND WINTER GESTERN loc-NORD SCHOTTLAND loc-REGION UEBERSCHWEMMUNG AMERIKA IX', ...
            self.glosses.append(gloss_ids)
            self.n_frames.append(anno['num_frames']) # 176, ...
            for i in range(anno['num_frames']):
                self.img_paths.append(f"{self.anno_path}@{anno['name']}.avi_pid0_fn{i:06d}-0.png") # 'data/phoenix-2014/phoenix-2014-videos.zip@fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png', ...

        if mode == 'train':
            pass
        elif mode == 'validation':
            self.data_transform = Compose([
                Resize(self.aug_size, interpolation='bilinear'),
                CenterCrop(size=target_size),
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
        # To-do: Load each video in the form of a zip file according to the number of clip_len.
        # Fetch video_name, gloss and frame count
        video_name = self.videos[idx]
        gloss_ids = self.glosses[idx]
        n_frames = self.n_frames[idx]

        # Get all the frames path for this videos 
        all_frame_paths = [
            path for path in self.img_paths if video_name in path
        ]
        
        # Select frames based on clip_len
        selected_frame_paths = self._sample_frames(all_frame_paths, n_frames)

        # # Load the frames as images
        # frames = [self.read_img(path) for path in selected_frame_paths]

        # Load frames dynamically from zip
        frames = [self.read_img_from_zip(path) for path in selected_frame_paths]
        """
        Question:

        Only in evaluation mode, we can use `data_transform` and `data_resize`.
        I think we should use rand-augment module in train mode. You can refer to `aug_frame` func below some classes

        Answer: Ah, this is just for testing whether the resizing and transform operations are working or not. Since there is no augmentation code in the __init__ code for training,
        I was confused what to do while in training. I was going to ask you about this. But now I will use something like _aug_frames functions in the classes below.
        """
        # Apply transformations
        if self.mode == 'train' and self.aug:
            frames = [self.data_transform(frame) for frame in frames]
        else:
            frames = [self.data_resize(frame) for frame in frames]
            frames = [self.data_transform(frame) for frame in frames]

        # Convert frames to tensor
        frames_tensor = torch.stack(frames, dim=0) # Shape: (clip_len, C, H, W)
        
        return frames_tensor, torch.tensor(gloss_ids, dtype=torch.long)

    """
    Question:

    This func takes a lot of times, How can we imporve this problem? Why don't you use `ZipReader` module?
    """
    def read_img_from_zip(self, path):
        """
        Read an image from the zip file dynamically.
        """
        try:
            # Extract the relative path inside the zip
            zip_path = path.split('@')[-1]  # Remove the prefix
            with zipfile.ZipFile(self.video_path, 'r') as zfile:
                with zfile.open(zip_path, 'r') as file:
                    image_data = file.read()
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    return image
        except Exception as e:
            raise RuntimeError(f"Failed to read image {path} from zip file: {e}")

        

    def _sample_frames(self, frame_paths, n_frames):
        """ Sample frames from a list of a frame paths. """
        if n_frames >= self.clip_len:
            return frame_paths[-self.clip_len:] 
        else:
            """
            Question:

            Why did you use `frame_paths[0]`? From what I understand, `frame_paths[0]` seems to represent
            the path to the first frame. In that case, shouldn't you return `padding + frame_paths` instead of
            `frame_paths + padding`? To return `frame_paths + padding`, wouldn't it make more sense to select the last frame? 
            """
            padding = [frame_paths[0]] * (self.clip_len - n_frames)
            return frame_paths + padding


if __name__ == "__main__":
    dataset = Phoenix2014Video(
        anno_path="/nas/Dataset/Phoenix/phoenix-2014.train",
        gloss_to_id_path="/nas/Dataset/Phoenix/gloss2ids.pkl",
        video_path="/nas/Dataset/Phoenix/phoenix-2014-videos.zip",
        mode="train",
        clip_len=256,
        target_size=(224, 224)
    )

    # Test data loading
    for i in range(2):  # Load two samples
        frames, gloss_ids = dataset[i]
        print(f"Sample {i+1}:")
        print(f"  Frames Shape: {frames.shape}")  # Should be (256, 3, 224, 224)
        print(f"  Gloss IDs: {gloss_ids}")       # List of gloss indices




# class CondensedMoviesVideo(Dataset):
#     def __init__(
#         self, anno_path: str, video_path: str='', split: str=',', mode: str='train', clip_len: int=16,
#         aug_size: tuple=(224, 224), target_size: tuple=(224, 224), trimmed: int=180, time_stride: int=60, args=None
#     ):
#         self.anno_path = anno_path # meta data path
#         self.video_path = video_path # video path
#         self.split = split # seperator in csv
#         self.mode = mode
#         self.clip_len = clip_len # n_frames
#         self.aug_size = aug_size # Augementation size
#         self.target_size = target_size # Input size to model
#         self.trimmed = trimmed # 사용할 비디오 길이
#         self.time_stride = time_stride
#         self.args = args
#         self.aug = False # 데이터 증강 여부

#         # 학습 과정시
#         if self.mode == 'train':
#             self.aug = True # 데이터 증강 사용
#             if self.args.reprob > 0.:
#                 self.rand_erase = True # random erasing 사용
        
#         import pandas as pd
#         import ast

#         df = pd.read_csv(self.anno_path, delimiter=self.split)
#         df.iloc[:, 1] = df.iloc[:, 1].apply(ast.literal_eval) # str to list

#         self.ori_ids = list(df.iloc[:, 0])
#         self.ori_labels = list(df.iloc[:, 1])
#         self.ori_durations = list(df.iloc[:, 2])

#         self.ids = []
#         self.labels = []
#         self.starts = []
#         self.ends = []

#         # 비디오 길이 조절
#         for idx, duration in enumerate(self.ori_durations):
#             if duration < trimmed:
#                 self.ids.append(self.ori_ids[idx])
#                 self.labels.append(self.ori_labels[idx])
#                 self.starts.append(0)
#                 self.ends.append(duration)
#             else:
#                 # 너무 긴 경우, 비디오를 잘라서 사용
#                 starts = [i for i in range(0, int(duration), time_stride)]

#                 # trimmed보다 긴 경우, time_stride 간격으로 비디오를 자름
#                 for start in starts:
#                     end = start + trimmed
#                     # 계산된 종료 시각이 실제 비디오 running time보다 길 경우
#                     if end > duration:
#                         if duration - start >= trimmed / 2: # 남은 비디오의 시간이 trimmed의 절반 이상인 경우
#                             end = duration
#                         else: # 그렇지 않으면 사용하지 않음
#                             continue
                    
#                     self.ids.append(self.ori_ids[idx])
#                     self.labels.append(self.ori_labels[idx])
#                     self.starts.append(start)
#                     self.ends.append(end)
        
#         # 개수가 많은 학습용 데이터만 제외해 전처리
#         if mode == 'train':
#             pass


#         elif mode == 'validation':
#             self.data_transform = Compose([
#                 Resize(self.aug_size, interpolation='bilinear'),
#                 CenterCrop(size=target_size),
#                 ClipToTensor(),
#                 Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]) # ImageNet
#             ])
#         elif mode == 'test':
#             self.data_resize = Compose([
#                 Resize(self.target_size, interpolation='bilinear')
#             ])
#             self.data_transform = Compose([
#                 ClipToTensor(),
#                 Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]) # ImageNet
#             ])

            
#     def __getitem__(self, idx: int):
#         if self.mode == 'train':
#             args = self.args
#             # Data informations
#             id = self.ids[idx]
#             start = self.starts[idx]
#             end = self.ends[idx]

#             buffer = self.loadvideo_decord(id, start, end, chunk_nb=-1) # (T, H, W, C) ndarray
            
#             if len(buffer) == 0:
#                 while len(buffer) == 0:
#                     warnings.warn("video {} not correctly loaded during training".format(id))
#                     index = np.random.randint(self.__len__())
#                     id = self.ids[index]
#                     start = self.starts[index]
#                     end = self.ends[index]
#                     buffer = self.loadvideo_decord(id, start, end, chunk_nb=-1)

#             # 증강 여부
#             if args.n_samples > 1: # n_samples: 증강 횟수
#                 frames = []
#                 labels = []
#                 indices = []
#                 # 데이터 증강
#                 for _ in range(args.n_samples):
#                     new_frames = self._aug_frame(buffer, args) # (C, T, H, W)
#                     frames.append(new_frames)
#                     labels.append(self.labels[idx])
#                     indices.append(idx)
                
#                 return frames, torch.tensor(labels, dtype=torch.float32), indices
#             else: # 한 번만 증강 수행
#                 buffer = self._aug_frame(buffer, args) # (C, T, H, W)
            
#             return buffer, torch.tensor(self.labels[idx], dtype=torch.float32), idx
        
#         elif self.mode == 'validation':
#             id = self.ids[idx]
#             start = self.starts[idx]
#             end = self.ends[idx]

#             buffer = self.loadvideo_decord(id, start, end, chunk_nb=0)

#             if len(buffer) == 0:
#                 while len(buffer) == 0:
#                     warnings.warn("video {} not correctly loaded during training".format(id))
#                     index = np.random.randint(self.__len__())
#                     id = self.ids[index]
#                     start = self.starts[index]
#                     end = self.ends[index]
#                     buffer = self.loadvideo_decord(id, start, end, chunk_nb=0)
            
#             # 입력을 위한 transform
#             buffer = self.data_transform(buffer) # Resize, CenterCrop, ClipToTensor, Normalize

#             return buffer, torch.tensor(self.labels[idx], dtype=torch.float32), id
        
#         elif self.mode == 'test':
#             id = self.ids[idx]
#             start = self.starts[idx]
#             end = self.ends[idx]

#             buffer = self.loadvideo_decord(id, start, end, chunk_nb=0)
            
#             while len(buffer) == 0:
#                 warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(id))
#                 idx = np.random.randint(self.__len__())
#                 sample = self.ids[idx]
#                 start = self.starts[idx]
#                 end = self.ends[idx]
#                 buffer = self.loadvideo_decord(sample, start, end, chunk_nb=0)

#             buffer = self.data_resize(buffer)
#             buffer = self.data_transform(buffer) # ClipToTensor, Normalize

#             return buffer, torch.tensor(self.labels[idx], dtype=torch.float32), id
        
#     def _aug_frame(self, buffer, args):
#         """
#         데이터 증강

#         buffer: Video tensor(T, H, W, C)
#         """
#         aug_transform = create_random_augment(
#             input_size=self.aug_size,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation
#         )
#         # 변환을 위해 Tensor -> Image
#         buffer = [transforms.ToPILImage()(frame) for frame in buffer] # (T, H, W, C)
#         buffer = aug_transform(buffer)

#         # 입력을 위해 Image -> Tensor
#         buffer = [transforms.ToTensor()(img) for img in buffer] # (T, C, H, W)
#         buffer = torch.stack(buffer) # (T, C, H, W)
#         # 정규화를 위해 shape 변형
#         buffer = buffer.permute(0, 2, 3, 1) # (T, H, W, C)
#         buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         # spatial sampling을 위해 shape 변형
#         buffer = buffer.permute(3, 0, 1, 2) # (C, T, H, W)

#         scl, asp = (
#             [0.5, 1.0],
#             [0.75, 1.3333],
#         )

#         buffer = spatial_sampling(
#             frames=buffer, target_height=self.target_size[0], target_width=self.target_size[1],
#             random_horizontal_flip=True, scale=scl, aspect_ratio=asp
#         )
        
#         if self.rand_erase:
#             random_erasing = RandomErasing(
#                 args.reprob, mode=args.remode, max_count=args.recount,
#                 num_splits=args.recount, device='cpu'
#             )
#             # random erasing을 위해 shape 변형
#             buffer = buffer.permute(1, 0, 2, 3) # (T, C, H, W)
#             buffer = random_erasing(buffer)
#             # 모델의 입력값으로 사용하기 위해 shape 변형
#             buffer = buffer.permute(1, 0, 2, 3) # (C, T, H, W)
        
#         return buffer
    
#     def _get_seq_frames(self, video_size: int, start: int, end: int, n_frames: int, clip_idx=-1):
#         """
#         비디오에서 프레임을 샘플링
        
#         video_size: 비디오의 프레임 수
#         start: 시작 프레임
#         end: 마지막 프레임
#         n_frames: 추출할 프레임 수
#         clip_idx: mode 확인
#         """
#         # 시작 및 끝 시간 확인
#         start = max(0, min(start, video_size - 1))
#         end = max(start, min(end, video_size - 1))

#         clipped_video_size = end - start + 1
#         seg_size = max(0., (clipped_video_size - 1) / n_frames) # segment 크기
#         seq = []

#         # 비디오를 같은 시간 간격으로 잘라 segment 생성
#         # 각 segment마다 랜덤으로 프레임을 샘플링
#         if clip_idx == -1: # Train
#             for i in range(n_frames):
#                 start_frame = int(np.round(seg_size * i)) + start # 각 segment의 첫 프레임
#                 end_frame = int(np.round(seg_size * (i + 1))) + start # 각 segment의 마지막 프레임
#                 idx = min(random.randint(start_frame, end_frame), end) # 각 segment에서 임의로 샘플링한 프레임
#                 seq.append(idx)
#         else: # Eval
#             n_seg = 1
#             # if self.mode == 'test':
#             #     n_seg = self.test_n_seg
#             duration = seg_size / (n_seg + 1)
#             for i in range(n_frames):
#                 start_frame = int(np.round(seg_size * i)) + start
#                 idx = min(start_frame + int(duration * (clip_idx + 1)), end) # 샘플링할 프레임
#                 seq.append(idx)
        
#         return seq



#     def loadvideo_decord(self, id: str, start: int, end: int, chunk_nb=0):
#         """
#         비디오를 ndarray 형태로 로드
        
#         id: Video id
#         start: 시작 시간
#         end: 끝 시간
#         chunk_nb: -1이면 학습 모드, 0이면 검증 모드, 그 이외의 정수 값은 테스트 모드에서의 chunk idx

#         Load video as ndarray

#         id: Video id
#         start: Start time
#         end: End time
#         chunk_nb: -1 is training mode, 0 is validation mode, other integer values ​​are chunk idx in test mode
#         """
#         f_name = os.path.join(self.video_path, id + '.mp4')
#         try:
#             vr = VideoReader(f_name, ctx=cpu(), num_threads=2)
#             fps = vr.get_avg_fps()
#             all_indices = self._get_seq_frames(len(vr), int(start * fps), int(end * fps), self.clip_len, clip_idx=chunk_nb)
            
#             while len(all_indices) < self.clip_len:
#                 print(f"Some frame is not loaded in video({id})")
#                 all_indices = self._get_seq_frames(len(vr), int(start * fps), int(end * fps), self.clip_len, clip_idx=chunk_nb)

#             vr.seek(0) # 시작 지점으로 읽는 지점 초기화

#             return vr.get_batch(all_indices).asnumpy() # (T, H, W, C)의 tensor로 반환
#         except:
#             print(f"Video cannot be loaded by decord: {f_name}")
#             return []

#     def __len__(self):
#         return len(self.ids)

# class CondensedMoviesText(Dataset):
#     def __init__(
#         self, anno_path: str, split: str=',', mode: str='train', max_len=0, tokenizer=None, args=None
#     ):
#         self.anno_path = anno_path # meta data path
#         self.split = split # seperator in csv
#         self.mode = mode
#         self.args = args

#         self.max_len = max_len
#         self.tokenizer = tokenizer
        
#         import pandas as pd
#         import ast

#         df = pd.read_csv(self.anno_path, delimiter=self.split)
#         df.iloc[:, 1] = df.iloc[:, 1].apply(ast.literal_eval) # str to list

#         self.ids = list(df.iloc[:, 0])
#         self.labels = list(df.iloc[:, 1])
#         self.durations = list(df.iloc[:, 2])
#         self.scripts = list(df.iloc[:, 3])
            
#     def __getitem__(self, idx: int):
#         script = self.scripts[idx]
#         id = self.ids[idx]
        
#         script = self.tokenizer.encode(
#             self.tokenizer.cls_token + self.tokenizer.bos_token + self.scripts[idx] + self.tokenizer.eos_token,
#             max_length=self.max_len, padding='max_length', truncation=True, add_special_tokens=False
#         )
        
#         return torch.tensor(script, dtype=torch.int32), torch.tensor(self.labels[idx], dtype=torch.float32), id
        
#     def __len__(self):
#         return len(self.ids)

# class CondensedMoviesAudio(Dataset):
#     def __init__(
#         self, anno_path: str, audio_path: str='', split: str=',', mode: str='train',
#         trimmed: int=180, time_stride: int=60, args=None
#     ):
#         self.anno_path = anno_path # meta data path
#         self.audio_path = audio_path # audio path
#         self.split = split # seperator in csv
#         self.mode = mode
#         self.trimmed = trimmed # 사용할 비디오 길이
#         self.time_stride = time_stride
#         self.args = args
        
#         import pandas as pd
#         import ast

#         df = pd.read_csv(self.anno_path, delimiter=self.split)
#         df.iloc[:, 1] = df.iloc[:, 1].apply(ast.literal_eval) # str to list

#         self.ori_ids = list(df.iloc[:, 0])
#         self.ori_labels = list(df.iloc[:, 1])
#         self.ori_durations = list(df.iloc[:, 2])

#         self.ids = []
#         self.labels = []
#         self.starts = []
#         self.ends = []

#         # 비디오 길이 조절
#         for idx, duration in enumerate(self.ori_durations):
#             if duration < trimmed:
#                 self.ids.append(self.ori_ids[idx])
#                 self.labels.append(self.ori_labels[idx])
#                 self.starts.append(0)
#                 self.ends.append(duration)
#             else:
#                 # 너무 긴 경우, 비디오를 잘라서 사용
#                 starts = [i for i in range(0, int(duration), time_stride)]

#                 # trimmed보다 긴 경우, time_stride 간격으로 비디오를 자름
#                 for start in starts:
#                     end = start + trimmed
#                     # 계산된 종료 시각이 실제 비디오 running time보다 길 경우
#                     if end > duration:
#                         if duration - start >= trimmed / 2: # 남은 비디오의 시간이 trimmed의 절반 이상인 경우
#                             end = duration
#                         else: # 그렇지 않으면 사용하지 않음
#                             continue
                    
#                     self.ids.append(self.ori_ids[idx])
#                     self.labels.append(self.ori_labels[idx])
#                     self.starts.append(start)
#                     self.ends.append(end)
        
#         # 개수가 많은 학습용 데이터만 제외해 전처리
#         if mode == 'train':
#             pass
            
            
#     def __getitem__(self, idx: int):
#         # Video informations
#         id = self.ids[idx]
#         start = self.starts[idx]
#         end = self.ends[idx]
        
#         audio = self.load_audio(id, start, end)
#         if len(audio) == 0:
#             while len(audio) == 0:
#                 warnings.warn("video {} not correctly loaded during training".format(id))
#                 index = np.random.randint(self.__len__())
#                 id = self.ids[index]
#                 start = self.starts[index]
#                 end = self.ends[index]
#                 audio = self.load_audio(id, start, end)
        
#         audio = torch.tensor(audio)
#         audio = tensor_normalize(audio, -35.533195, 23.727068)

#         return audio, torch.tensor(self.labels[idx], dtype=torch.float32), idx

#     def load_audio(self, id: str, start: int, end: int):
#         """
#         오디오를 ndarray 형태로 로드
        
#         id: Video id
#         start: 시작 시간
#         end: 끝 시간
#         """
#         f_name = os.path.join(self.audio_path, id + '.npy')
#         hop_length = 0.01
#         pad_width = ((0, 0), (0, int(self.trimmed / hop_length - (end - start) / hop_length)))
#         try:
#             audio = np.load(f_name)[:, int(start / hop_length):int(end / hop_length)]
#             audio = np.pad(audio, pad_width, mode='constant')
            
#             while audio.shape[1] != int(self.trimmed / hop_length):
#                 audio = np.pad(audio, ((0, 0), (0, int(self.trimmed / hop_length) - audio.shape[1])), mode='constant')
            
#             return audio
#         except:
#             print(f"Audio cannot be loaded: {f_name}")
            
#             return []
    
#     def __len__(self):
#         return len(self.ids)

# def spatial_sampling(
#         frames: torch.Tensor, target_height: int=224, target_width: int=224,
#         random_horizontal_flip: bool=True, scale: Optional[list]=None, aspect_ratio: Optional[list]=None
#     ):
#     """
#     Random resized crop -> Horizontal flip
#     """
#     transform_func = random_resized_crop_with_shift
#     frames = transform_func(
#         images=frames, target_height=target_height, target_width=target_width, scale=scale, ratio=aspect_ratio
#     )
#     if random_horizontal_flip:
#         frames, _ = horizontal_flip(.5, frames)
    
#     return frames

# def tensor_normalize(tensor: torch.Tensor, mean, std):
#     tensor = tensor.float()
#     if type(mean) == list:
#         mean = torch.tensor(mean)
#     if type(std) == list:
#         std = torch.tensor(std)
#     tensor = tensor - mean
#     tensor = tensor / std
#     return tensor

# def build_dataset(modal, is_train, is_test, args):
#     assert modal in ('video', 'keypoint')
#     print(f"Use Dataset: {args.dataset}")

#     if args.dataset == 'phoenix-2014':
#         mode = None
#         anno_path = None

#         if modal == 'video':
#             if is_train == True:
#                 mode = 'train'
#                 anno_path = os.path.join(args.metadata_path, 'phoenix-2014.train')
#             elif is_test == True:
#                 mode = 'test'
#                 anno_path = os.path.join(args.metadata_path, 'phoenix-2014.test')
#             else:
#                 mode = 'validation'
#                 anno_path = os.path.join(args.metadata_path, 'phoenix-2014.dev')
        
#             dataset = Phoenix2014Video(
#                 anno_path=anno_path, video_path=args.video_path, mode=mode, clip_len=args.n_frames,
#                 aug_size=args.aug_size, target_size=args.video_size, args=args
#             )
#         elif modal == 'text':
#             if is_train == True:
#                 mode = 'train'
#                 anno_path = os.path.join(args.metadata_path, 'mm_train.csv')
#             elif is_test == True:
#                 mode = 'test'
#                 anno_path = os.path.join(args.metadata_path, 'mm_test.csv')
#             else:
#                 mode = 'validation'
#                 anno_path = os.path.join(args.metadata_path, 'mm_val.csv')
        
#             dataset = CondensedMoviesText(
#                 anno_path=anno_path, split=args.split, mode=mode, max_len=args.max_len,
#                 tokenizer=args.tokenizer, args=args
#             )
#         elif modal == 'audio':
#             if is_train == True:
#                 mode = 'train'
#                 anno_path = os.path.join(args.metadata_path, 'train.csv')
#             elif is_test == True:
#                 mode = 'test'
#                 anno_path = os.path.join(args.metadata_path, 'test.csv')
#             else:
#                 mode = 'validation'
#                 anno_path = os.path.join(args.metadata_path, 'val.csv')
        
#             dataset = CondensedMoviesAudio(
#                 anno_path=anno_path, audio_path=args.audio_path, split=args.split, mode=mode,
#                 trimmed=args.trimmed, time_stride=args.time_stride, args=args
#             )
#         # For original
#         all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
#         cls_cnt_list = torch.tensor(np.asarray(dataset.labels).sum(axis=0), device=args.device)
#     else:
#         print(f"Wrong: {args.data_set}")
#         raise NotImplementedError
    
#     assert len(all_genres) == args.n_classes
#     print(f"# of classes: {len(all_genres)}")

#     return dataset, all_genres, cls_cnt_list