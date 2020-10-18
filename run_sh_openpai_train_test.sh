echo NAIC Online Score Reproduction of DMT

# echo Train
# python train.py --config_file configs/naic_2020_split.yml
# echo Test
# python test.py --config_file configs/naic_2020_split.yml
# echo evaluation
# python ./mics/eval.py
echo Train and Eval
python train_val.py --config_file configs/naic_2020_split.yml