import torch
from torch.utils.data import DataLoader
from utils import collate_fn  # collate_fn이 정의된 모듈 경로를 수정하세요.
from datasets.dataset import build_dataset  # build_dataset 함수가 정의된 모듈 경로를 수정하세요.

# 설정값 지정
dataset_config = {
    "metadata_path": '/nas/Dataset/Phoenix',  # 어노테이션 경로
    "gloss_to_id_path": '/nas/Dataset/Phoenix/gloss2ids.pkll',  # 글로스 ID 경로
    "video_path": '/nas/Dataset/Phoenix/phoenix-2014-videos.zip',  # 비디오 경로
    "mode": 'train',  # 'validation', 'test' 가능
    "clip_len": 256,
    "aug_size": (210, 210),
    "target_size": (224, 224),
}

batch_size = 4  # 배치 크기 설정

# 데이터셋 빌드
print("Building dataset...")
dataset = build_dataset(
    anno_path=dataset_config["anno_path"],
    gloss_to_id_path=dataset_config["gloss_to_id_path"],
    video_path=dataset_config["video_path"],
    mode=dataset_config["mode"],
    clip_len=dataset_config["clip_len"],
    aug_size=dataset_config["aug_size"],
    target_size=dataset_config["target_size"],
    args=dataset_config["args"]
)

# 데이터셋 길이 확인
print(f"Dataset length: {len(dataset)}")

# 단일 샘플 확인
sample = dataset[0]
frames, gloss_ids, idx = sample
print(f"\nSingle sample:")
print(f" - Frames shape: {frames.shape} (C, T, H, W)")
print(f" - Gloss IDs: {gloss_ids}")
print(f" - Sample index: {idx}")

# DataLoader를 사용한 배치 테스트
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print("\nTesting DataLoader with collate_fn:")
for batch in dataloader:
    frames_batch, gloss_ids_batch, indices = batch
    print(f"Batch Frames shape: {frames_batch.shape} (B, C, T_max, H, W)")
    print(f"Batch Gloss IDs shape: {gloss_ids_batch.shape} (B, N_max)")
    print(f"Batch Indices: {indices}")
    break  # 한 배치만 출력

# 이미지 읽기 테스트
sample_path = dataset.img_paths[0]
image = dataset.read_img(sample_path)
print(f"\nRead Image shape: {image.shape} (H, W, C)")

# 데이터 어그멘테이션 테스트 (훈련 모드에서만 가능)
if dataset_config["mode"] == 'train':
    augmented_frames = dataset._aug_frame([image], args=None)
    print(f"\nAugmented Frames shape: {augmented_frames.shape} (C, T, H, W)")

# 글로스 토큰화 테스트
gloss_seq = dataset.glosses[0]
tokenized_gloss = dataset.gloss_tokenizer([gloss_seq])
print("\nTokenized Gloss IDs:")
print(tokenized_gloss['labels'][0])
