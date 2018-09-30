path = "C:/Users/guobaba/Desktop/rnn_prediction.txt"

with open(path,"rb") as f:
    doc= f.readlines()
    num =list(map(lambda c:int(str(c).split(r"\t")[0].split("'")[1]),doc))
    i=280001
    for o in num:
        if i!=o:
            print(i,o)
            break
        i+=1

