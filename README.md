# GRUPoetry
Code for [*Generating Chinese Classical Poems with RNN Encoder-Decoder*](https://link.springer.com/chapter/10.1007/978-3-319-69005-6_18) (CCL 2017). More related resources are available at [THUAIPoet](https://github.com/THUNLP-AIPoet).

## 0. Rights ##
All rights reserved.

## 1. Basic Model ##

We built our poetry generator with GroundHog, more details of the Gated Unit RNN model please refer to <https://github.com/lisa-groundhog/GroundHog>

## 2. Setting ##

All needed source codes are in the directory GroundHogAttention. We did a few changes on the source codes, thus they are not totally same as the original GroundHog.

Please add GroundHogAttention to python path, then preprocess your poetry corpus with the tools in GroundHogAttention/experiments/nmt/preprocess. More details please refere to <https://github.com/lisa-groundhog/GroundHog>.

You can put your training data (mainly the vocabulary) , and the model file, e.g. search_model.npz and the config file, e.g. search_state.pkl in GroundHogAttention/experiments/nmt/SPB/ and run generate.py to generate poetry lines.

## 3. Other Blocks ##
Other Blocks described in our paper, such as CPB and WPB, can be easily got by changing the training data.

## 4. System
This work has been integrated into the automatic poetry generation system, ** THUAIPoet(Jiuge, 九歌)**, which is available via https://jiuge.thunlp.cn.  This system is developed by Research Center for Natural Language Processing, Computational Humanities and Social Sciences, Tsinghua University (清华大学人工智能研究院与社会人文计算研究中心). Please refer to [THUAIPoet](https://github.com/THUNLP-AIPoet), [THUNLP](https://github.com/thunlp) and [THUNLP Lab](http://nlp.csai.tsinghua.edu.cn/site2/) for more information.

## 5. Cite ##
Xiaoyuan Yi, Ruoyu Li, and Maosong Sun. 2017. Generating Chinese Classical Poems with RNN Encoder-Decoder. In Proceedings of the Sixteenth Chinese Computational Linguistics, pages 211–223, Nanjing, China

## 6. Contact ##
If you have any questions, suggestions and bug reports, please email yi-xy16@mails.tsinghua.edu.cn or mtmoonyi@gmail.com.
