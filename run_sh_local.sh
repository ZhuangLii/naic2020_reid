echo NAIC Online Score Reproduction of DMT



# python divided_dataset.py \
# --data_dir_query /mnt/nfs-internstorage/train_data/naic2020_evaldataset/test/query_a \
# --data_dir_gallery /mnt/nfs-internstorage/train_data/naic2020_evaldataset/test/gallery_a \
# --save_dir /mnt/nfs-internstorage/train_data/naic2020_evaldataset/test

# python divided_dataset.py --data_dir_query ../data/naic_mannual_dataset/test/query_a --data_dir_gallery ../data/naic_mannual_dataset/test/gallery_a --save_dir ../data/naic_mannual_dataset/test


# python divided_dataset.py --data_dir_query /mnt/data/zhuangjunfei/NAIC_handcraft_2020/query_a --data_dir_gallery /mnt/data/zhuangjunfei/NAIC_handcraft_2020/gallery_a --save_dir /mnt/data/zhuangjunfei/NAIC_handcraft_2020/

echo remove

rm /home/zjf/naic_code/log/19_20_a_auto/*

echo Train
python train_val.py --config_file configs/naic_round2_model_a_local.yml

# echo Test
# here we will get Distmat Matrix after test.
# python test_local.py --config_file configs/naic_round2_model_a_local.yml

# echo evalutaion
# python ./mics/eval_local.py 