# GRUPoetry
Chinese Classical Poetry Generation with Gated Unit RNN. The implementation details of this project please see <http://arxiv.org/abs/1604.01537>

----------------------------------

## 1. Basic Model ##

We built our poetry generator with GroundHog, more details of the Gated Unit RNN model please refer to <https://github.com/lisa-groundhog/GroundHog>

## 2. Setting ##

All needed source codes are in the directory GroundHogAttention. We did a few changes on the source codes, thus they are not totally same as the original GroundHog.

Please add GroundHogAttention to python path, then preprocess your poetry corpus with the tools in GroundHogAttention/experiments/nmt/preprocess, more details please refere to <https://github.com/lisa-groundhog/GroundHog>.

You can put your training data (mainly the vocabulary) , and the model file, e.g. search_model.npz and the config file, e.g. search_state.pkl in GroundHogAttention/experiments/nmt/SPB/ and run generate.py to generate poetry lines.