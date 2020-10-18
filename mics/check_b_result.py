import json
from glob import glob
from tqdm import tqdm

g_root = '/mnt/data/naic_2020_B/test/gallery_a'
q_root = '/mnt/data/naic_2020_B/test/query_a'
g_images = glob(g_root + '/*.png')
q_images = glob(q_root + '/*.png')

g_images = [x.split('/')[-1] for x in g_images]
q_images = [x.split('/')[-1] for x in q_images]

json_path = '/mnt/nfs-internstorage/user/zjf/NAIC2020/NAIC_2020_B_results/final/three_ensemble.json'
json_path2 = '/mnt/nfs-internstorage/user/zjf/NAIC2020/NAIC_2020_B_results/final/19_20_a_auto_598_rerakningpso_model_test_on_B.json'
with open(json_path, 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
with open(json_path2, 'r', encoding='utf8') as fp:
    json_data2 = json.load(fp)

for k, v in tqdm(json_data.items()):
    assert k in q_images 
    q_images.remove(k)

    assert len(v) == 200    
    for m in v:
        assert v.count(m) == 1
        assert m in g_images

assert len(q_images) == 0

for k,v in tqdm(json_data2.items()):
    try:
        assert k in list(json_data.keys())
    except:
        print(k)
for k,v in tqdm(json_data.items()):
    try:
        assert k in list(json_data2.keys())
    except:
        print(k)
