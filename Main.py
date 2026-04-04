import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

#데이터 불러오기
data = load_breast_cancer()

input = data.data
target = data.target


