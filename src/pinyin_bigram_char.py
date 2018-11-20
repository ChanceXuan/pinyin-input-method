import os
import sys
import pickle
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

def data_processing(char, file):
    # 计算一二级汉字表中出现了多少种单个汉字
    print('开始生成汉字表元素索引……')
    char_list = list()
    for i in char:
        if (char.find(i)!=-1):
            char_list.append(char.find(i))
    print('汉字表元素索引生成完毕。')

    # 把文件的每个字转换为一二级汉字表中该字的索引值，存储到列表
    print('开始将训练语料库以汉字索引值的形式表示……')
    file_list = list()
    for i in file:
        if (char.find(i)!=-1):
            file_list.append(char.find(i))
        else:
            file_list.append(-1)
    print('训练语料库汉索引值化完成。')   
    return char_list, file_list

def p_cal(char_list, file_list):
    n = len(char_list)
    l = len(file_list)
    # 计算单字、两字共现的概率
    freq_char = np.ones(n)
    freq_matrix = np.ones((n,n))

    p_char = np.zeros(n)
    p_matrix = np.zeros((n,n))

    print("开始计算单字、两字共现的概率……")
    if (file_list[0] != -1):
        freq_char[file_list[0]] += 1
    for i in range(1,l):                        # i指文件中每个字在一二级汉字表中的索引值
        if (file_list[i] != -1):
            freq_char[file_list[i]] += 1        # 存储文件中第i个字对应的index在单字概率相同index下的频数
            if ( file_list[i-1] != -1):
                freq_matrix[file_list[i-1],file_list[i]] += 1
                                                # 存储文件中第i个字和前一个字对应的index在联合概率矩阵中分别相同index下的频数
    print("概率计算完毕")

    # 计算方案1: 先每行求和得出每个字的频数，再（共现概率/每个字的频数）
    freq_matrix_sum = np.sum(freq_matrix, axis=1)
    p_matrix = freq_matrix/np.array([freq_matrix_sum]).T
    # p_char = freq_char/freq_matrix_sum

    # 计算方案2: 直接（共现概率/每个字的频数）
    # p_matrix = freq_matrix/np.array([freq_char]).T
    p_char = freq_char/np.sum(freq_char)

    return p_char, p_matrix

def dict_gen(char, table):
    table = table.splitlines()          # a list which contains each row as an element

    pinyin_list = list()                # a list which contains each pinyin as a sub-list
    for i in range(0,len(table)):
        pinyin_list.append((table[i]).split(" "))

    pinyin_dict = {}                    # a dictionaty which contains each pinyin as a key, along with index of characters
    for i in range(0,len(pinyin_list)):
        pinyin_dict[pinyin_list[i][0]] = list()
        for j in range(1,len(pinyin_list[i])):
            pinyin_dict[pinyin_list[i][0]].append(char.find(pinyin_list[i][j]))
    return pinyin_dict

def dp(input, pinyin_dict, p_char, p_matrix):
    # input指第i句话，以列表的形式存放了这句话中的每个字作为元素
    row = 0
    print(input)
    for column in range(0,len(input)):
        if row < len(pinyin_dict[input[column]]):
            row = len(pinyin_dict[input[column]])
    # 先列后行
    p_point = np.zeros((len(input),row))        # 存储每一个点的可能性信息
    relation = np.zeros((len(input),row))       # 存储上一个相关点的行信息
    open_dict = {}
    
    # 第一个拼音对应汉字索引的循环
    for row in range(0,len(pinyin_dict[input[0]])):
        p_point[0,row] = p_char[pinyin_dict[input[0]][row]]   # 由拼音的单字概率确定的每一个汉字的联合概率
        tup = (0,row)                                         # 找出单字在整句话各个拼音对应汉字的表格中的位置
        open_dict[tup] = p_point[0,row]                       # 待扩展点由{(位置)：概率}组成
    # print('首字概率计算完成')

    # 之后开始用联合概率
    k = 0
    while len(open_dict)!=0:
        prior_node = {(0,0):0}
        key_prior_node = list(prior_node)[0]

        # 找出open_dict里的优先扩展节点
        for i in open_dict:
            # print('当前元组为：',i)
            if (open_dict[i] > prior_node[key_prior_node]):
                prior_node = {i:open_dict[i]}
                key_prior_node = list(prior_node)[0]
        k = k+1
        row_prior_node = key_prior_node[1]
        column = key_prior_node[0]
        index_key_prior_node = pinyin_dict[input[column]][row_prior_node]
        # print(k,'轮比较后，当前优先节点为：',prior_node)

        if (column == len(input)-1):
            open_dict.clear()
            # print('结束计算过程')
            break

        else:   # 提取优先扩展节点在整句话中的位置，以元组形式存储             
            # print('开始扩展优先节点……')
            # 下一个拼音的每一个汉字索引选项的循环
            for row in range(0,len(pinyin_dict[input[column+1]])):
                index_key_expand_node = pinyin_dict[input[column+1]][row]

                # 由这个拼音确定的下一个拼音的每一种汉字情况的联合概率
                p_temp = p_point[column,row_prior_node]*p_matrix[index_key_prior_node][index_key_expand_node]

                if p_temp > p_point[column+1,row]:
                    p_point[column+1,row] = p_temp
                    tup = (column+1,row)
                    open_dict[tup] = p_point[column+1][row]
                    relation[column+1][row] = row_prior_node
            del open_dict[key_prior_node]
            # print('结束本次扩展。')

    output_list = []
    output_list.append(pinyin_dict[input[column]][row_prior_node])
    row_back_node = row_prior_node

    for column in range(len(input)-1,0,-1):
        row_back_node = int(relation[column,row_back_node])
        output_list.insert(0,pinyin_dict[input[column-1]][row_back_node])
    return output_list

# 主程序运行
if (len(sys.argv) == 1):
    print ("Input from keyboard!")
    while True:
        input = [input()]
        break

if (len(sys.argv) == 2):
    try:
        input = open(sys.argv[1], "r", encoding = "utf-8").read().splitlines()
    except:
        print ('No such file!\n')
        exit()
# input = open('../data/input_validation.txt').read().splitlines()
input_list = list()
for i in range(0,len(input)):
    input_list.append((input[i]).split(" "))
print('\n输入拼音为：')

char_open = open('../resource/一二级汉字表.txt',encoding='gbk').read()       
dict_open = open("../resource/拼音汉字表.txt",encoding = 'gbk').read()
pinyin_dict = dict_gen(char_open, dict_open)

if os.path.exists("../data/p_char.pkl"):
    with open("../data/p_char.pkl","rb") as f:
        p_char = pickle.load(f)
    with open("../data/p_matrix.pkl","rb") as f:
        p_matrix = pickle.load(f)

else:
    file_open = open('../resource/sina_news_gbk/2016-11.txt',encoding='gbk').read()\
                + open('../resource/sina_news_gbk/2016-10.txt',encoding='gbk').read()\
                + open('../resource/sina_news_gbk/2016-09.txt',encoding='gbk').read()\
    dict_open = open("../resource/拼音汉字表.txt",encoding = 'gbk').read()

    # 进行数据处理，得出汉字表索引，以及用汉字索引值表示的文字
    (char_processed,file_processed) = data_processing(char_open,file_open)

    # 计算单字、两字共现的概率，用矩阵表示
    (p_char,p_matrix) = p_cal(char_processed,file_processed)
    with open("../data/p_char.pkl","wb") as f:
        pickle.dump(p_char,f)
    with open("../data/p_matrix.pkl","wb") as f:
        pickle.dump(p_matrix,f)

output_list = list()
for i in range(0,len(input_list)):
    sentence = dp(input_list[i], pinyin_dict, p_char, p_matrix)
    output_list.append(sentence)

with open("../data/output.txt","w") as f:
    f.write('输出结果为：\n')

print('\n输出结果为：')
for i in range(0,len(output_list)):
    sentence = ''
    for j in output_list[i]:
        sentence = sentence + char_open[j]
    print(sentence)
    with open("../data/output.txt","a") as f:
        f.write(sentence + '\n')