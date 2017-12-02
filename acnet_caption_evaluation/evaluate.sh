#!/usr/bin/env bash

# Convert input json files to submission format.
rm -rf tmp/*
python format_convert/convert_with_gt_proposals.py input tmp

# Evaluate.
cd densevid_eval
for input_file in ../tmp/*
do
    output_file=../output/$(basename $input_file .json).result
    python evaluate.py \
    -s $input_file \
    -o $output_file \
    --verbose
done
