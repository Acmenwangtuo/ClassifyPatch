import pandas as pd 
import os
import json


def read(csv_path):
    data_type = {'id':str,'label':int}
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
    cnt = 0
    for k in tabel.keys():
        data = [i for i in data_list if i[0].split('-')[2] == k]
        img_list = []
        diag_num = 0
        sus_num = 0 
        for value in data:
            if value[1] == 1:
                diag_num += 1
                img_list.append(value[0])
            elif value[1] == 2:
                sus_num += 1
        normal_num = len(data) - diag_num - sus_num
        # if diag_num >= 1:
        #     diag_class = 'Diagnosis'
        #     cnt += 1
        # elif normal_num > sus_num:
        #     diag_class = 'Normal'
        # else:
        #     diag_class = 'Suspicious'
        max_value = max(sus_num,diag_num,normal_num)
        if max_value == diag_num:
            diag_class = 'Diagnosis'
            cnt += 1
        elif max_value == normal_num:
            diag_class = 'Normal'
        else:
            diag_class = 'Suspicious'
        tabel[k]['img_list'] = img_list
        tabel[k]['class'] = diag_class
        tabel[k]['diag_num'] = diag_num
        tabel[k]['normal_num'] = normal_num
        tabel[k]['sus_num'] = sus_num
        tabel[k]['all_num'] = len(data)
    json_str = json.dumps(tabel,indent=4)
    with open('./diag_result_three.json','w') as json_file:
        json_file.write(json_str)
    print(cnt,len(tabel.keys()))

def main():
    res = read('./three_diag/res.csv')
    # print(res[0])
    # print(res)
    group(res)

if __name__ == '__main__':
    main()
    
