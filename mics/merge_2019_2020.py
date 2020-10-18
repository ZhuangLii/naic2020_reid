'''
合并2019和2020 naic 数据集，把 2019 train文件夹下的图片加入 2020 train 下面, 使用此脚本合并label
'''

label_2020 = '/mnt/data/zhuangjunfei/NAIC_DATA_2020/train/label.txt'
label_2019 = '/mnt/data/zhuangjunfei/NAIC_DATA_2019/2019_fusai/2019_label.txt'
output_label = './2019_2020_merge_label.txt'
with open(label_2020, 'r') as f:
    labels_2020 = f.readlines()
    labels_2020 = ['2020_'+x for x in labels_2020]
with open(label_2019, 'r') as f:
    labels_2019 = f.readlines()
with open(output_label, 'w') as f:
    all_labels = []
    all_labels.extend(labels_2020)
    pids_2020 = set()
    for item in labels_2020:
        pids_2020.add(item.rstrip().split(':')[1])
    # check 
    for i in range(len(pids_2020)):
        assert str(i) in pids_2020, 'person id error'
    for item in labels_2019:
        img, pid = item.rstrip().split(' ')
        img = '2019_' + img.replace('train/','')
        new_pid = len(pids_2020) + int(pid)
        all_labels.append(img+':' + str(new_pid)+'\n')
    for item in all_labels:
        f.write(item)        

