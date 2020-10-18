echo NAIC Online Score Reproduction of DMT

echo Train

python divided_dataset.py --data_dir_query ./data/test/query_a --data_dir_gallery ./data/test/gallery_a --save_dir ./data/test/

python train.py --config_file configs/naic_round2_model_a.yml

echo Test
# here we will get Distmat Matrix after test.
python test.py --config_file configs/naic_round2_model_a.yml

# python test.py --config_file configs/naic_round2_model_b.yml

# python test.py --config_file configs/naic_round2_model_se.yml

echo evaluation

# python ./mics/eval.py