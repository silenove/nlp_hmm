#coding=utf-8

import re
import numpy as np
import codecs
from sklearn.model_selection import KFold

#读入语料
data_read = codecs.open('raw_data.txt', 'r', 'utf-8')
raw_data = data_read.readlines()
raw_data_join = ' '.join(raw_data)

sentences = []
for raw_sentence in raw_data:
    sens = re.split('[，。]+', raw_sentence)
    sentences.extend(sens)
sentences = np.array(sentences)


#状态集合
pattern = re.compile(r'/[a-z]+[0-9]*')
match = pattern.findall(raw_data_join)
match = list(set(match))
match.sort()

state = {}
n = 0
for m in match:
    state[m] = n
    n += 1

#观察集合
items = raw_data_join.split(' ')
words = []
for item in items:
    s = item.strip('\n').split('/')
    if len(s[0]) != 0 and s[-1] != 'w':
        words.append('/'.join(s[0:-1]))
words = list(set(words))
words.sort()
words.append('unknown')
observe = {}
n = 0
for word in words:
    observe[word] = n
    n += 1

#计算状态集合、观察集合的维度
dim_state = len(state)
dim_observe = len(observe)

#将句子分为词项
def sentence_split(sentence):
    subs = sentence.split(' ')
    words = []

    for sub in subs:
        s = sub.strip('\n').split('/')
        if len(s[0]) != 0 and s[-1] != 'w':
            s_state = '/' + s[-1]
            s_observe = '/'.join(s[0:-1])
            words.append((s_state, s_observe))
    return words

def get_key(dic, value):
    q = [k for k, v in dic.items() if v == value]
    if len(q) > 0:
        return q[0]
    else:
        return -1

# 对新样本进行预测，输入sentence为[word, word, ……]
def predict(sentence, S, A, B, state, observe):  # S为初始状态概率，A为状态转移概率矩阵，B为观察概率矩阵，state为状态集合，observe为观察集合

    length = len(sentence)
    dim_state = len(state)
    state_pre = []  # 预测状态序列

    un = 'unknown'  # 未统计词项

    # 记录路径矩阵
    trans = np.matrix(np.zeros((dim_state, length)) - np.ones((dim_state, length)))

    if length < 1:
        return state_pre

    last = np.array(np.zeros(len(state)))
    for i in range(len(subs)):
        tmp = np.array(np.zeros(len(state)))
        if subs[i][1] in observe.keys():  # 统计过的词项
            if i == 0:
                for j in range(len(state)):
                    tmp[j] = S[j] * B[j, observe[sentence[i]]]
            else:
                for j in range(len(state)):
                    for k in range(len(state) - 1):
                        if tmp[j] < last[k] * A[k, j] * B[j, observe[sentence[i]]]:
                            tmp[j] = last[k] * A[k, j] * B[j, observe[sentence[i]]]
                            trans[j, i] = k
        else:  # 未统计过的词项
            if i == 0:
                for j in range(len(state)):
                    tmp[j] = S[j] * B[j, observe[un]]
            else:
                for j in range(len(state)):
                    for k in range(len(state)):
                        if tmp[j] < last[k] * A[k, j] * B[j, observe[un]]:
                            tmp[j] = last[k] * A[k, j] * B[j, observe[un]]
                            trans[j, i] = k
        last = np.array(tmp)

    # 回溯记录状态路径
    max_prob = 0
    max_index = -1
    for i in range(len(last)):
        if last[i] > max_prob:
            max_index = i
            max_prob = last[i]
    state_pre.append(get_key(state, max_index))

    state_recall = max_index
    for i in range(1, length):
        state_pre.append(get_key(state, trans[state_recall, length - i]))
        state_recall = int(trans[state_recall, length - i])

    state_pre.reverse()
    return state_pre


# 交叉验证
split_rate = 0.8
print('train : ', int(len(sentences)*0.8))
print('test : ' , int(len(sentences)*0.2))

kf = KFold(n_splits=5)
epoch = 1
for train_index, test_index in kf.split(sentences):

    print('epoch : ', epoch)
    epoch += 1

    train = sentences[train_index]
    test = sentences[test_index]

    # 初始状态概率
    state_start = np.array(np.zeros(dim_state))
    # 状态转移矩阵
    state_trans = np.matrix(np.zeros((dim_state, dim_state))) + 0.001

    # 观察概率矩阵
    observe_prob = np.matrix(np.zeros((dim_state, dim_observe))) + 0.001
    # 根据句子统计
    for sentence in train:
        subs = sentence_split(sentence)

        if len(subs) < 1:
            continue

        # 统计开始状态频率
        if subs[0][0] in state.keys():
            state_start[state[subs[0][0]]] += 1

        # 统计状态转移频率
        for i in range(0, len(subs) - 1):
            s_state, t_state = subs[i][0], subs[i + 1][0]
            if s_state in state.keys() and t_state in state.keys():
                state_trans[state[s_state], state[t_state]] += 1

        # 统计观察频率
        for i in range(0, len(subs)):
            s_state = subs[i][0]
            s_observe = subs[i][1]
            if s_state in state.keys() and s_observe in observe.keys():
                observe_prob[state[s_state], observe[s_observe]] += 1

    # 计算概率
    sum_start = sum(state_start)
    state_start = state_start / sum_start

    for i in range(state_trans.shape[0]):
        if state_trans[i].sum() != 0:
            state_trans[i] = state_trans[i] / state_trans[i].sum()

    for i in range(observe_prob.shape[0]):
        if observe_prob[i].sum() != 0:
            observe_prob[i] = observe_prob[i] / observe_prob[i].sum()

    #在测试集上评估
    acc = 0
    num = 0
    test_total = len(test)
    for sentence in test:
        sen_acc = 0
        subs = sentence_split(sentence)
        sentence_words = [w for s, w in subs]
        state_pre = predict(sentence_words, state_start, state_trans, observe_prob, state, observe)

        if len(state_pre) > 0:
            for i in range(min(len(subs), len(state_pre))):
                if subs[i][0] == state_pre[i]:
                    sen_acc += 1
            acc += sen_acc / len(state_pre)
        else:
            acc += 1

        num += 1

        if num % 100 == 0:
            print((num / test_total)*100 , '%' +' finished, acc : ' ,  acc / num)

    acc = acc / test_total
    print('test acc : ', acc)
