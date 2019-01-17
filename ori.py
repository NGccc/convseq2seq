f = open('c2oric.txt','r',encoding='utf-8-sig')
txt=f.read()
dic=eval(txt)
print(dic)
i2s = {}
for k,v in dic.items():
    i2s[str(v)] = k

print(i2s)

f.close()

f = open('./data/labels_c.txt','r',encoding='utf-8-sig')
fw = open('./data/data.label_c','w',encoding='utf-8-sig')
lines = f.readlines()
for line in lines:
    l = ''
    ts = line.split()
    #import pdb
    #pdb.set_trace()
    for i in ts:
        #print(i)
        l = l + i2s[i] + ' '
    fw.write(l+'\n')
    #print(l)
f.close()
fw.close()