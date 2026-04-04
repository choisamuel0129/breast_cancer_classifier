import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

#데이터 불러오기
data = load_breast_cancer()
print(data.data.shape)

inputs = data.data
target = data.target

#데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(inputs, target, stratify=target, test_size=0.2) #훈련데이터와 테스트+검증 데이터 분리
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, stratify=train_target, test_size=0.2)#테스트데이터와 검증데이터 분리
