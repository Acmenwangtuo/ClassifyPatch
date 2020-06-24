import pandas as pd 
import os
import json


def read(csv_path):
    data_type = {'id':str,'label':float}
    data = pd.read_csv(csv_path,dtype=data_type)
    print(data)
    res = []
    for idx,row in data.iterrows():
        res.append([row['id'],row['label']])
    return res
    
def group(data_list):
    tabel = {}
    for k,v in data_list:
        if tabel.get(k.split('-')[2],None) is None:
            tabel[k.split('-')[2]] = {}
        else:
            continue
    for k in tabel.keys():
        data = [i for i in data_list if i[0].split('-')[2] == k]
        img_list = []
        diag_num = 0
        for value in data:
            if value[1] >= 0.5:
                diag_num += 1
                img_list.append(value[0])
        if diag_num >= 5:
            diag_class = 'Diagnosis'
        else:
            diag_class = 'Normoal'
        tabel[k]['img_list'] = img_list
        tabel[k]['class'] = diag_class
        tabel[k]['diag_num'] = diag_num
        tabel[k]['normal_num'] = len(data) - diag_num
        tabel[k]['all_num'] = len(data)
    json_str = json.dumps(tabel,indent=4)
    with open('./normal_reslut.json','w') as json_file:
        json_file.write(json_str)

def main():
    res = read('../er_all_nonpositive/res.csv')
    # print(res[0])
    # print(res)
    group(res)

if __name__ == '__main__':
    main()
    
