#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
rm -rf summary_dir/*
rm -rf train_dir/*
python train_lstm_decoder.py \
--training_data_loader data_loader_data/data_loader_train_inception.dat \
--validation_data_loader data_loader_data/data_loader_val1_inception.dat \
--vocab_file data/vocab.txt \
--train_dir train_dir \
--summary_dir summary_dir \
--decoder_version Scratch
