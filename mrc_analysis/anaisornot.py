import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# 列名： alternatives, passage, query, answer, query_id, url
VALID_SET_FILE = "../mrc_data/ai_challenger_oqmrc_validationset.json"
TRAIN_SET_FILE = "../mrc_data/ai_challenger_oqmrc_trainingset.json"
TEST_SET_FILE = "../mrc_data/ai_challenger_oqmrc_testa.json"

mrc = pd.DataFrame(data=pd.read_json(open(TEST_SET_FILE, encoding='utf-8'), encoding='utf-8', lines=True))

# 删除无用列
mrc.drop('url', axis=1, inplace=True)
mrc.drop('query_id', axis=1, inplace=True)

# 增加统计列
mrc['passage_len'] = mrc['passage'].map(len)
mrc['qyr_len'] = mrc['query'].map(len)
mrc['alternatives_re'] = mrc['alternatives'].str.replace("无法", "")

# print(mrc['qyr_len'].value_counts())
'''
Valid
            passage_len         qyr_len
count       30000.000000        30000.000000
mean        87.061033           10.677900
std         106.708837          3.304499
min         19.000000           5.000000
25%         37.000000           9.000000
50%         60.000000           10.000000
75%         102.000000          12.000000
max         5923.000000         44.000000
valid un isornot item total is 1485
valid al both in passage 485
 valid  one or tow  in passage  1000
 valid only one in passage  515
 valid only one in passage acc  454
 valid ACC 88.15%
 train un isornot item total is 11802
 train al both in passage 4031
 train al one or tow in passage 8226
 train only one in passage 4191
 train only one in passage acc 3729
 train ACC 88.97%
 test  only one in passage 454

'''

pattern6 = r"不|对|没|假|无|否"   # 选项中包含没|不|假|无|对|否 238142


kki=mrc[~mrc['alternatives_re'].str.contains(pattern6) & (mrc['passage_len'] < 500)].filter(['passage', 'alternatives']).values
print("非是非题"+str(len(kki)))
#
# al one or tow in passage
# re=[]
# for k in kki :
#     p = k[0]
#     al = str(k[1]).split("|")
#     for i in al:
#         if "无法" not in i:
#             # if (al[0] in p) | (al[1] in p) | (al[2] in p):
#             if i in p:
#                 re.append(str(al)+str(p))
#                 break;
#
# print(len(re))
# print(re)

#************************************************************
#al both in passage
# re=[]
# for k in kki :
#     p = k[0]
#     al = str(k[1]).split("|")
#     for i in al :
#         if "无法" in i :
#             al.remove(i)
#     if len(al)<2:
#         continue
#     if(al[0] in p )& (al[1] in p):
#         re.append(str(al)+str(p))
#
# print(len(re))
# print(re)

# ————————————————————————————————————————

# al only one in passage
re=[]
for k in kki :
    p = k[0]
    al = str(k[1]).split("|")
    pl=str(k[1]).split("|")
    for i in al :
        if "无法" in i :
            al.remove(i)
    if len(al)<2:
        continue
    if((al[0] in p )& (al[1] not in p))|((al[0] not in p )& (al[1] in p)):
        # if "无法" not in pl[0]:
        re.append(str(pl)+str(p))

print(len(re))
print(re)
