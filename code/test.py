import torch
from model import Model

if __name__ == '__main__':
    model = Model()
    dummy_frames = torch.rand(2, 3, 2, 720, 1280)
    model(dummy_frames, 'd')