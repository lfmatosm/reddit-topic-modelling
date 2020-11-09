#!/bin/bash

folder=$1
pt_dataset=$2
pt_dataset_name=$3
en_dataset=$4
en_dataset_name=$5
field="body"

echo "Processando dataset original em portugues..."
python3 text-preprocessor/preprocess.py \
    --datasetFile $pt_dataset \
    --datasetName  $pt_dataset_name \
    --datasetFolder $folder \
    --field $field \
    --lang pt
echo "Dataset original em portugues processado"

echo "Processando dataset original em ingles..."
python3 text-preprocessor/preprocess.py \
    --datasetFile $en_dataset \
    --datasetName  $en_dataset_name \
    --datasetFolder $folder \
    --field $field \
    --lang en
echo "Dataset original em ingles processado"
