import random
import math
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
import matplotlib
import matplotlib.pyplot as plt

g_dataset = {}
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter
KNN_NEIGHBOR = 7


##########################
##### Load Data  #########
##########################

# Convert next digit from input file as a vector
# Return (digit, vector) or (-1, '') on end of file
def read_digit(p_fp):  # 读单个数据
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS+1))
    if bits == '':
        return -1, bits
    # convert bit string as digit vector
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec


def read_pre_digit(p_fp):  # 读预测数据单个数据
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS+1))
    if bits == '':
        return -1, bits, None
    # convert bit string as digit vector
    narray, temp = [], []
    for i in bits:
        if i != '\n':
            temp.append(int(i))
        else:
            narray.append(temp)
            temp = []

    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec, narray


# Parse all digits from training file
# and store all digits (as vectors)
# in dictionary g_dataset
def load_data(p_filename=DATA_TRAINING):  # 读取训练集
    global g_dataset
    # Initial each key as empty list
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)
    return g_dataset


def load_testdata(p_filename=DATA_TESTING):  # 读取测试集
    dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            dataset[val].append(vec)
    return dataset


def load_predict_data(p_filename=DATA_PREDICT):  # 读取预测数据
    dataset = []
    with open(p_filename) as f:
        while True:
            val, vec, narray = read_pre_digit(f)
            if val == -1:
                break
            dataset.append([vec, numpy.array(narray)])
    return dataset


def shuffle_datas(dataset):
    total_dataset = []
    for i in range(10):
        for j in dataset[i]:
            total_dataset.append([j, i])
    random.shuffle(total_dataset)
    return total_dataset


##########################
##### kNN Models #########
##########################

# Given a digit vector, returns
# the k nearest neighbor by vector distance
def knn(p_v, train, size=KNN_NEIGHBOR, datanum=500):
    nn = []
    for vector, d in train:
        dist = round(distance(p_v, vector), 2)
        nn.append((dist, d))

    nn.sort(key=lambda x: x[0])

    # TODO: find the nearest neigbhors

    return nn[:size+1]


# Based on the knn Model (nearest neighhor),
# return the target value
def knn_by_most_common(p_v, train, data_num=500):
    nn = knn(p_v, train[:data_num])

    # TODO: target value
    return nn[0][1]


##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model
# Parse each digit from the predict file
# and print the predicted balue
def predict(train, p_filename=DATA_PREDICT):
    # TODO
    # print('TO DO: show results of prediction')
    pre_dataset = load_predict_data()
    sample = random.sample(pre_dataset, 1)
    answer = knn_by_most_common(sample[0][0], train)
    print(f"预测结果为:{answer}")
    font = {'color': 'red', 'size': 20}
    plt.imshow(sample[0][1], cmap='gray')
    plt.title(f"KNN prediction result:{answer}", fontdict=font)
    plt.show()


##########################
##### Vector     #########
##########################

# Return distance between vectors v & w
def distance(v, w):
    gap_sum = 0
    for i in range(len(v)):
        gap = v[i]-w[i]
        gap_sum += gap**2
    dis = math.sqrt(gap_sum)
    return dis


if __name__ == '__main__':
    load_data()
    total_train_dataset = shuffle_datas(g_dataset)  # 我把原来数据集的那个二维改了一下结构，现在是[[向量, 标签]...]，这个shuffle_datas是打乱数据集顺序
    predict(total_train_dataset)  # 预测
    """
    核心函数是knn_by_most_common, knn, distance
    """
    # show_info()
    # validate()
    # predict()
