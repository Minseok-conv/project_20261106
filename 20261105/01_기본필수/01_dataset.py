"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

torch.set_num_threads(1)
torch.manual_seed(14)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# transforms.Compose()는 이미지에 적용할 여러 변환을 순서대로 묶어주는 함수.

train = ImageFolder(DATA / "train") 

print("class_to_idx:", train.class_to_idx)

# ImageFolder를 사용하려면 기본적으로 이미지들이 클래스별 하위 폴더에 나뉘어 있어야함
# class_to_index는 ImageFolder객체가 가지고 있는 속성임.
# ImageFolder가 하위 폴더 이름을 클래스 이름으로 인식하고, 기본적으로 알파벳 순으로 정렬해 인덱스를 부여함.

for split in ["train", "valid", "test"]:
    dataset = ImageFolder(DATA / split)
    counts = [dataset.targets.count(i) for i in range(3)]
    #dataset.targets는 ImageFolder가 지정된 루트 폴더 아래에서 찾은 모든 이미지의 클래스 인덱스(라벨)를 리스트로 반환
    print(split, "개수:", counts, "같은 매핑:", dataset.class_to_idx == train.class_to_idx)
# test는 파일명·라벨 매핑만 확인했습니다. 모델 예측에는 사용하지 않습니다.
