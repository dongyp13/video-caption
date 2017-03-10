This package contains the codes for the following paper:

* \[1\] Yinpeng Dong, Hang Su, Jun Zhu, Bo Zhang. Improve interpretabilirt of Deep Neural Networks with Semantic Information (In preparation). CVPR 2017.

[Video](http://ml.cs.tsinghua.edu.cn/~yinpeng/papers/demo-cvpr17.mp4)

The codes are forked from [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid).

#####We illustrate the running details, which can also be found in their repo. (We make a little change)

Note: due to the fact that video captioning research has gradually converged to using [coco-caption](https://github.com/tylin/coco-caption) as the standard toolbox for evaluation. We intergrate this into this package. In the paper, however, a different tokenization methods was used, and the results from this package is *not* strictly comparable with the one reported in the paper. 

#####Please follow the instructions below to run this package
1. Dependencies
  1. [Theano](http://deeplearning.net/software/theano/) can be easily installed by following the instructions there. Theano has its own dependencies as well. The simpliest way to install Theano is to install [Anaconda](https://store.continuum.io/cshop/anaconda/). Instead of using Theano coming with Anaconda, we suggest running `git clone git://github.com/Theano/Theano.git` to get the most recent version of Theano. 
  2. [coco-caption](https://github.com/tylin/coco-caption). Install it by simply adding it into your `$PYTHONPATH`.
  3. [Jobman](http://deeplearning.net/software/jobman/install.html). After it has been git cloned, please add it into `$PYTHONPATH` as well. 
2. Download the preprocessed version of Youtube2Text. It is a zip file that contains everything needed to train the model. Unzip it somewhere. By default, unzip will create a folder `youtube2text_iccv15` that contains 8 `pkl` files. 
[preprocessed YouTube2Text download link](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/yaoli/youtube2text_iccv15.zip)

3. Go to `common.py` and change the following two line `RAB_DATASET_BASE_PATH = '/home/yinpeng/mfs/video-caption/dataset/'` and `/home/yinpeng/mfs/video-caption/result/'` according to your specific setup. The first path is the parent dir path containing `youtube2text_iccv15` dataset folder. The second path specifies where you would like to save all the experimental results.
4. Copy `attribute.pkl` to `youtube2text_iccv15` folder. `attribute.pkl` contains the topic representations for each video described in our paper. We use [WarpLDA](https://github.com/thu-ml/warplda) to extract the topic representations. You can also define your own topics.

5. Before training the model, we suggest to test `data_engine.py` by running `python data_engine.py` without any error.
6. It is also useful to verify coco-caption evaluation pipeline works properly by running `python metrics.py` without any error.
7. Now ready to launch the training
  1. to run on cpu: `THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python train_model.py`
  2. to run on gpu: `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_model.py`

#####Notes on running experiments
Running `train_model.py` for the first time takes much longer since Theano needs to compile for the first time lots of things and cache on disk for the future runs. You will probably see some warning messages on stdout. It is safe to ignore all of them. Both model parameters and configurations are saved (the saving path is printed out on stdout, easy to find). The most important thing to monitor is `train_valid_test.txt` in the exp output folder. It is a big table saving all metrics per validation. Please refer to `model_attention.py` line 1207 -- 1215 for actual meaning of columns. 


#####Bonus
In the paper, we never mentioned the use of uni-directional/bi-directional LSTMs to encode video representations. But this is an obvious extension. In fact, there has been some work related to it in several other recent papers following ours. So we provide codes for more sophicated encoders as well. 

#####Trouble shooting
This is a known problem in COCO evaluation script (their code) where METEOR are computed by creating another subprocess, which does not get killed automatically. As METEOR is called more and more, it eats up mem gradually. 
To fix the problem, add this line after line https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/meteor.py#L44
`self.meteor_p.kill()`

If you have any questions, drop us email at li.yao@umontreal.ca.

