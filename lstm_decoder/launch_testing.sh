#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="3,4,5"
python test_lstm_decoder.py \
--vocab_file data/vocab.txt \
--validation_data_loader data_loader_data/data_loader_train_inception.dat \
--model_path models/tmp_4/model-1000 \
--output_file output.json
