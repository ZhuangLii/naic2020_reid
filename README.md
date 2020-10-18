# naic2020_reid
naic 2020 AI+ 行人重识别项目


## 方案说明

### 建立验证集
使用 `./divided_dataset.py` 统计NAIC2020 reid的数据集，包括`train, test/gallery, test/query`中的green以及normal image 数量

normal image
![green image](./pic/green.jpg)

green image

![normal2 image](./pic/normal2.jpg)


|  gallery_green_num   | gallery_normal_num  | query_green_num | query_normal_num | train_green_num | train_normal_num
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
| 4149  | 	36317 | 1376  | 1524 | 39446  | 33378 |

其中，gallery_normal_num > train_normal_num, 因此不能从train 数据集中提取 normal image来建立 val 数据集。我们从测试集中提取 normal image 放入 val 的 gallery中充当干扰数据。具体code在`./mics/make_eval_dataset.py`。

### 具体策略

在 [naic2020示例程序](https://naic.pcl.ac.cn/frame/3) 的基础上, 我们从以下几个方面进行了测试
| 数据处理                              | 数据增强                | backbone         | loss                    | solver                        | attention        |
|-----------------------------------|---------------------|------------------|-------------------------|-------------------------------|------------------|
| 去除长尾数据                            | Mixup               | se\-resnet101\-a | CurricularFace          | SGD                           | selective kernel |
| naic mean std 替换imagenet mean std | random patch        | efficientnet     | TripletLoss with margin | centralized\_gradient\_SGD    | cbam             |
|                                   | big size 384 \* 192 | densenet         | AMSoftmax               | centralized\_gradient\_Ranger | non\_local       |
|                                   | AutoAugment         | inceptionv4      |                         |                               |                  |

其中比较多无效的尝试，下面我们仅列出有效的方法在A榜中的得分比较。（其中baseline 代表naic2020示例程序中的参数配置）

* baseline 0.468
* 加入niac 2019 数据集到train数据集，且去掉center loss 0.558
* 替换baseline中的数据增广策略为AutoAugment 0.598
* reranking parameter使用粒子群优化搜索超参数 0.611

最终，在B榜采用的方案为上述三种方法的综合： 即加入2019数据集 + AutoAugment + PSO reranking, 模型存放在`./ckpt/resnet101_ibn_a.pth`, 使用的参数配置文件`./configs/naic_round2_model_a.yml`。

### 复现
训练model
```
mv path/to/naic2019/train_dataset/images/*.png ./data/train/
mv path/to/naic2020/train_dataset/images/*.png ./data/train/
mv path/to/test/query/*.png ./data/test/query_a/
mv path/to/test/gallery/*.png ./data/test/gallery_a/
mv ./mics/2019_2020_merge_label.txt ./data/train/label.txt
python 
```
