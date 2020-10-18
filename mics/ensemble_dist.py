import numpy as np
import os
import json
from os.path import join


def worker3(dir1, dir2, dir3, output):
    query_1 = np.load(dir1 + '1229-a-1/query_path_1.npy')
    gallery_1 = np.load(dir1 + '1229-a-1/gallery_path_1.npy')
    query_2 = np.load(dir1 + '1229-a-1/query_path_2.npy')
    gallery_2 = np.load(dir1 + '1229-a-1/gallery_path_2.npy')

    distmat_1 = np.load(dir1 + '1229-a-1/distmat_1.npy')
    distmat_1 += np.load(dir2 + '1229-a-1/distmat_1.npy')
    distmat_1 += np.load(dir3 + '1229-a-1/distmat_1.npy')

    indexes = np.argsort(distmat_1, axis=1)

    res_1 = {}
    for idx, index in enumerate(indexes):
        query = os.path.basename(query_1[idx])
        gallery = [os.path.basename(i) for i in gallery_1[index][:200].tolist()]
        res_1[query] = gallery

    distmat_2 = np.load(dir1 + '1229-a-1/distmat_2.npy')
    distmat_2 += np.load(dir2 + '1229-a-1/distmat_2.npy')
    distmat_2 += np.load(dir3 + '1229-a-1/distmat_2.npy')

    indexes = np.argsort(distmat_2, axis=1)

    res_2 = {}
    for idx, index in enumerate(indexes):
        query = os.path.basename(query_2[idx])
        gallery = [os.path.basename(i) for i in gallery_2[index][:200].tolist()]
        res_2[query] = gallery

    data = dict()
    for k, v in res_1.items():
        data[k] = v
    for k, v in res_2.items():
        data[k] = v

    save_path = 'submit_final.json'
    if not os.path.exists(output):
        os.mkdir(output)
    print("Writing to {}".format(join(output,save_path)))
    json.dump(data, open(join(output, save_path), 'w'))


def worker2(dir1, dir2, output):
    query_1 = np.load(dir1 + '1229-a-1/query_path_1.npy')
    gallery_1 = np.load(dir1 + '1229-a-1/gallery_path_1.npy')
    query_2 = np.load(dir1 + '1229-a-1/query_path_2.npy')
    gallery_2 = np.load(dir1 + '1229-a-1/gallery_path_2.npy')

    distmat_1 = np.load(dir1 + '1229-a-1/distmat_1.npy')
    distmat_1 += np.load(dir2 + '1229-a-1/distmat_1.npy')

    indexes = np.argsort(distmat_1, axis=1)

    res_1 = {}
    for idx, index in enumerate(indexes):
        query = os.path.basename(query_1[idx])
        gallery = [os.path.basename(i)
                   for i in gallery_1[index][:200].tolist()]
        res_1[query] = gallery

    distmat_2 = np.load(dir1 + '1229-a-1/distmat_2.npy')
    distmat_2 += np.load(dir2 + '1229-a-1/distmat_2.npy')

    indexes = np.argsort(distmat_2, axis=1)

    res_2 = {}
    for idx, index in enumerate(indexes):
        query = os.path.basename(query_2[idx])
        gallery = [os.path.basename(i)
                   for i in gallery_2[index][:200].tolist()]
        res_2[query] = gallery

    data = dict()
    for k, v in res_1.items():
        data[k] = v
    for k, v in res_2.items():
        data[k] = v

    save_path = 'submit_final.json'
    if not os.path.exists(output):
        os.mkdir(output)
    print("Writing to {}".format(join(output, save_path)))
    json.dump(data, open(join(output, save_path), 'w'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemable results")
    parser.add_argument("--ensemble_number", "-en", type=int, default=2)
    parser.add_argument(
        "--flag1", "-f1", default="19_20_densenet50_triplet_neckfeat_log")
    parser.add_argument(
        "--flag2", "-f2",  type=str, default="19_20_efficientb5_triplet_neckfeat_rerank_log")
    parser.add_argument(
        "--flag3", "-f3", type=str, default="19_20_triplet_0.55779264_log")
    args = parser.parse_args()
    flag1 = args.flag1
    flag2 = args.flag2
    flag3 = args.flag3
    # base = '/mnt/nfs-internstorage/user/zjf/NAIC2020'
    base = '/mnt/nfs-internstorage/user/zjf/NAIC2020/NAIC_2020_B_results/distmats'
    if args.ensemble_number == 3:
        dir1 = join(base, flag1) + '/'
        dir2 = join(base, flag2) + '/'
        dir3 = join(base, flag3) + '/'
        output = join(base, 'ensemble_results', flag1+flag2+flag3) 
        worker3(dir1, dir2, dir3, output)
    elif args.ensemble_number == 2:
        dir1 = join(base, flag1) + '/'
        dir2 = join(base, flag2) + '/'
        output = join(base, 'ensemble_results', flag1+flag2)
        worker2(dir1, dir2, output)

