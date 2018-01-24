# TSCN

This repository contains the code for **Temporal Segment Captioning Network**(TSCN). The paper can be found in the repository.


## Results

The models are evaluated on [ActivityNet Dense Captioning Dataset](http://activity-net.org/challenges/2017/captioning.html) with the ground truth video segment information.

|    Models   | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | CIDER |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:-----:|
|   LSTM-YT   |  18.22 |  7.43  |  3.24  |  1.24  |  6.56  | 14.86 |
|     S2VT    |  20.35 |  8.99  |  4.60  |  2.62  |  7.85  | 20.97 |
|    H-RNN    |  19.46 |  8.78  |  4.34  |  2.53  |  8.02  | 20.18 |
|   C3D+LSTM  |  26.45 |  13.48 |  7.12  |  3.98  |  9.46  | 24.56 |
| TSCN (ours) |  18.28 |  7.32  |  2.93  |  1.22  |  9.07  | **25.62** |

Note: The best version of TSCN found so far is defined in `lstm_decoder_scratch.py`. The corresponding hyper-parameters can be found [here](https://github.com/YuMao1993/ActivityNetVideoCaptioning/blob/a6baa483c7b01b8134795b098c1c04a550e5c80e/lstm_decoder/lstm_decoder_config.py).

## Training

+ Setting up the hyper-parameters in `lstm_decoder_config.py`.
+ Execute `launch_training_*.sh`.

## Testing

+ Setting up the hyper-parameters in `lstm_decoder_config.py`.
	+ Please make sure that the hyper-parameters used for testing and training are the same, otherwise the model will fail to load.
+ Execute `launch_testing_*.sh`.

## Evaluation

+ After testing, you should get a `*.json` result file (the path of the result file is set in your `launch_testing_*.sh` script).
+ Clean `acnet_caption_evaluation/input` folder.
+ Copy the result `*.json` file into `acnet_caption_evaluation/input`.
+ Under directory `acnet_caption_evaluation`. Execute `evaluate.sh`. The final scores will be dumped in `acnet_caption_evaluation/output`.

## Notice

+ To Chunjia and Harsha: Please make a copy of the repository on lab machine before doing any experiments so that we don't mess up each other's modification.


## Code Structure

+ **deep\_features** code for extracting deep network features from RGB image frames.
+ **lstm\_decoder** LSTM decoders, training and testing scripts.
	+ **Data Preparation and Loading**
		+ `vocabulary_builder.py`: Script for building vocabulary from ActivityNet caption JSON file.
		+ `build_vocabulary_from_training_set.sh`: Launching script for `vocabulary_builder.py`.
		+ `data_loader.py` Data loader that provides functionalities for loading trianing or validation deep features, checking validity, sampling mini-batches, etc.
		+ `build_data_loader.py` Script for building a serialized offline data loader so the data loading can be done in memory (mainly for acceleration).
	+ **Tensorflow Model Definitions**
		+ `lstm_decoder.py`: The first version of TSCN, which is a naive implementation of TSCN following show and tell paper.
		+ `lstm_decoder_repeated_feed_image.py`: The second version of TSCN based on `lstm_decoder.py`. In this version, we concatenate the the video feature embedding with the input word embedding and feed them as input at every time steps. In the original show-and-tell model, the image feature embedding is only fed as input at the first time step.
		+ `lstm_decoder_scratch.py`: The third version of TSCN based on `lstm_decoder_repeated_feed_image.py`. This version implements LSTM from scratch (instead of using TensorFlow's LSTM function). Also, the initial cell states and hidden states are treated as learnable parameters. Only one LSTM layer is used. Empirically, learning the initial state plays an important role in boosting the performance. **This version is best TSCN model found so far.**
		+ `lstm_decoder_attention.py`: The fourth version of TSCN based on `lstm_decoder_scratch.py`. The main modification is adding the temporal attention mechanism. A context vector is constructed by weightedly combining frame-level feature vectors (annotation vectors) of the sampled frames. The blending weights are computed from the hidden state (which encodes the information about the
context) and the annotation vectors.

	+ **Training Routine**
		+ `train_lstm_decoder.py`: Main training routine that is shared by all LSTM models.
		+ `launch_training_*.sh`: Launching scripts for training different LSTM models.

	+ **Inference Routine**
		+ `test_lstm_decoder.py`: Main testing routine that is shared by all LSTM models.
		+ `launch_testing_*.sh`: Launching scripts for testing different LSTM models.

	+ **Hyper-parameter Configurations**
		+ `lstm_decoder_config.py`: Hyper-parameters of the architecture and training protocol. Note: Despite that fact that we have different LSTM models, they all share the same configuration file. This is not very convenient at the moment. In the future, each model will have a separate configuration file. 
		+ `const_config.py`: Configuration parameters that should not be changed. 

+ **acnet\_caption\_evaluation**: Code for evaluating different scores of the result. See `Evaluation` section on how to use it.
+ **i3d**: model definition and training script of I3D.
+ **nn_captioning** Please ignore.
+ **utils**: Please ignore.
+ **tool_scripts**: Please ignore.

## References


[1] [Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

[2] [Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International Conference on Machine Learning. 2015.](http://proceedings.mlr.press/v37/xuc15.pdf)

[3] [Krishna, Ranjay, et al. "Dense-Captioning Events in Videos." arXiv preprint arXiv:1705.00754 (2017).](http://cs.stanford.edu/people/ranjaykrishna/densevid/)


## Contact

Yu Mao (yumao@cmu.edu)

