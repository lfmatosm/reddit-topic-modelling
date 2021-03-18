#!/bin/bash

train_batch_name=$1
preprocessed_dataset=$2
dictionary_file=$3
embeddings_file=$4
word_lemma_maps=$5
train_size=$6

echo -e "\n*************************************************************************************"

echo -e "Preparation: $train_batch_name"

echo -e "\nPreparing training resources (datasets, vocabulary, dictionary, embedding, etc)..."

python preparation/prepare_training_resources.py \
    --dataset $preprocessed_dataset --dataset_name $train_batch_name \
    --dictionary $dictionary_file \
    --embeddings $embeddings_file \
    --word_lemma_maps $word_lemma_maps \
    --train_size $train_size || exit 1

echo -e "\nTraining resources prepared"

base_prepared_resources_dir="resources/$train_batch_name"

echo -e "\nCopying original preprocessed dataset to '$base_prepared_resources_dir' folder..."
cp $preprocessed_dataset $base_prepared_resources_dir
echo -e "\nMoved dataset to '$base_prepared_resources_dir' folder"

echo -e "\nCopying dictionary to '$base_prepared_resources_dir' folder..."
cp $dictionary_file $base_prepared_resources_dir
echo -e "\nMoved dictionary to '$base_prepared_resources_dir' folder"

echo -e "\n*************************************************************************************"
