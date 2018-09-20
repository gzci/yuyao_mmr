import pandas as pd
import re
# 列名： alternatives, passage, query, answer, query_id, url
VALID_SET_FILE = "./ai_challenger_oqmrc_validationset.json"
TRAIN_SET_FILE = "./ai_challenger_oqmrc_trainingset.json"
TEST_SET_FILE = "./ai_challenger_oqmrc_testa.json"
pattern7 = r"无法"
mrc = pd.DataFrame(data=pd.read_json(open(TRAIN_SET_FILE, encoding='utf-8'), lines=True))
rr = re.compile(r"不|对|没|假|无|否")
unn=[]
uncount=[]
for line in mrc["alternatives"]:
    out = ["", "", ""]
    if len(rr.findall(str(line).replace("无法",""))) > 0:
        linetr = line.strip().split("|")

        for i in linetr:
            if "无法" in i:
                out[2]=i
            elif len(rr.findall(i)) > 0:
                out[1]=i
            else:
                out[0]=i
        print(str(out[0]+'|'+out[1]+'|'+out[2]))
        unn.append(str(out[0]+'|'+out[1]+'|'+out[2]))
    else:
        print("非是非题")
        unline= line.strip().split("|")
        for i in unline:
            if "无法" in i:
                out[2]=i
                unline.remove(i)
        if len(unline)>=2:

            if len(uncount)<12500/2 :
                out[0]=unline[0]
                out[1] = unline[1]
            else:
                out[0] = unline[1]
                out[1] = unline[0]
        else:
            out[0]=out[2]
            out[1]=out[2]
        print(str(out[0] + '|' + out[1] + '|' + out[2]))
        unn.append(str(out[0] + '|' + out[1] + '|' + out[2]))
        uncount.append(1)



print(len(unn))
print(unn)
mrc.iloc[:,0]=unn
# print(mrc[0])
with open('newtrain.json', 'w', encoding='utf-8') as file:
    mrc.to_json(file, force_ascii=False,orient="records")
# mrc.to_json("./newvaild.json",orient="records")
