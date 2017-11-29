#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="6,7,8,9,10,11,12"
rm -rf summary_dir/*
rm -rf train_dir/*
python train_lstm_decoder.py \
--training_data_loader data_loader_data/data_loader_train_inception.dat \
--validation_data_loader data_loader_data/data_loader_val1_inception.dat \
--vocab_file data/vocab.txt \
--train_dir train_dir \
--summary_dir summary_dir