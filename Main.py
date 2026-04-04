import keras

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

#데이터 불러오기
data = load_breast_cancer()
print(data.data.shape)

inputs = data.data
target = data.target

#데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(inputs, target, stratify=target, test_size=0.2) #훈련데이터와 테스트데이터 분리
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, stratify=train_target, test_size=0.2)# 훈련 데이터를 다시 훈련 데이터와 검증 데이터로 분리

#표준화
scaler = StandardScaler()
scaler.fit(train_input)
train_scaled = scaler.transform(train_input)
val_scaled = scaler.transform(val_input)
test_scaled = scaler.transform(test_input)

#로지스틱 회귀
def logistic_model():
    sc = SGDClassifier(loss='log_loss')
    sc.fit(train_scaled, train_target)
    print("훈련데이터성능: ", sc.score(train_scaled, train_target))
    print("테스트데이터성능: ", sc.score(test_scaled, test_target))

logistic_model()

#얕은 MLP
def MLP():
    input = keras.layers.Input(shape=(30,)) #입력층 생성
    dense = keras.layers.Dense(1, activation='sigmoid')
    model = keras.Sequential([input, dense])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_scaled, train_target, epochs=5)
    print("검증데이터성능: ", model.evaluate(val_scaled, val_target))
    print("테스트데이터성능: ", model.evaluate(test_scaled, test_target))

MLP()

#깊은 MLP
def MLPDense():
    input = keras.layers.Input(shape=(30,))
    dense1 = keras.layers.Dense(100, activation='relu')
    dense2 = keras.layers.Dense(30, activation='relu')
    dropout = keras.layers.Dropout(0.3)
    dense3 = keras.layers.Dense(1, activation='sigmoid')

    model = keras.Sequential([input, dense1, dense2, dropout, dense3])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_scaled, train_target, epochs=5)
    print("검증데이터성능: ", model.evaluate(val_scaled, val_target))
    print("테스트데이터성능: ", model.evaluate(test_scaled, test_target))

MLPDense()

#깊은 MLP + 체크포인트와 조기 종료
def MLPBest():
    input = keras.layers.Input(shape=(30,))
    dense1 = keras.layers.Dense(100, activation='relu')
    dense2 = keras.layers.Dense(30, activation='relu')
    dropout = keras.layers.Dropout(0.3)
    dense3 = keras.layers.Dense(1, activation='sigmoid')

    model = keras.Sequential([input, dense1, dense2, dropout, dense3])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
    print("검증데이터성능: ", model.evaluate(val_scaled, val_target))
    print("테스트데이터성능: ", model.evaluate(test_scaled, test_target))

MLPBest()