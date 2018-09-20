import pandas as pd
import re
# 列名： alternatives, passage, query, answer, query_id, url
VALID_SET_FILE = "./ai_challenger_oqmrc_validationset.json"
TRAIN_SET_FILE = "./ai_challenger_oqmrc_trainingset.json"
TEST_SET_FILE = "./ai_challenger_oqmrc_testa.json"
mrc = pd.DataFrame(data=pd.read_json(open(VALID_SET_FILE, encoding='utf-8'), lines=True))
print(mrc)

# f =open("./newvaild.json","r")
# p = f.readlines()
# for i in p:
#     str(i)
#     print(i)
#     print(i.encode("gbk").decode("gbk"))