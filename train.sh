#!/bin/bash

train_batch_name=$1
lang=$2

# K
topics="5 8 10 12 15 18 20 22 25 28 30"

echo -e "\n*************************************************************************************"

echo -e "Training: $train_batch_name"

base_prepared_resources_dir="resources/$train_batch_name"

echo -e "\nStarting ctm, lda and etm training..."

# CTM
echo -e "\nStarting ctm training...\n"
python training/ctm.py \
    --train_documents $base_prepared_resources_dir/train_documents.json \
    --validation_documents $base_prepared_resources_dir/validation_documents.json \
    --data_preparation $base_prepared_resources_dir/ctm_data_preparation.obj \
    --prepared_training_dataset $base_prepared_resources_dir/ctm_training_dataset.dataset \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --lang $lang \
    --dataset_name $train_batch_name \
    --topics $topics || exit 1

# LDA
echo -e "\nStarting lda training...\n"
python training/lda.py \
    --train_documents $base_prepared_resources_dir/train_documents.json \
    --validation_documents $base_prepared_resources_dir/validation_documents.json \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --lang $lang \
    --dataset_name $train_batch_name \
    --topics $topics || exit 1

# # ETM
echo -e "\nStarting etm training...\n"
python training/etm.py \
    --train_documents $base_prepared_resources_dir/train_documents.json \
    --validation_documents $base_prepared_resources_dir/validation_documents.json \
    --training_dataset $base_prepared_resources_dir/etm_training_dataset.dataset \
    --vocabulary $base_prepared_resources_dir/etm_vocabulary.vocab \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --embeddings $base_prepared_resources_dir/etm_w2v_embeddings.w2v \
    --lang $lang \
    --dataset_name $train_batch_name \
    --topics $topics || exit 1

echo -e "\nTraining finished successfully"

notebook_name="$(date '+%Y-%m-%d')_$train_batch_name"
notebook_extension=".ipynb"
evaluation_base_path="evaluation/$notebook_name"

echo -e "\nCreating evaluation folder at '$evaluation_base_path' and moving training outputs to the folder..."
mkdir -p $evaluation_base_path/resources
cp -R training_outputs/csvs $evaluation_base_path
cp -R training_outputs/models $evaluation_base_path
cp -r pipeline_logs/. $evaluation_base_path/logs
cp -r $base_prepared_resources_dir/. $evaluation_base_path/resources
rm -rf training_outputs
rm -rf $base_prepared_resources_dir

echo -e "\nCreating notebook at '$evaluation_base_path$notebook_extension'..."
cp -R evaluation_example/utils $evaluation_base_path
cp evaluation_example/evaluation_example.ipynb $evaluation_base_path/$notebook_name$notebook_extension

echo -e "\nCleaning generated files..."
rm -rf training_outputs
rm -rf pipeline_logs
rm -rf $base_prepared_resources_dir
echo -e "\nFolder cleaned"

echo -e "\nPostprocessing finished successfully"

echo -e "\n*************************************************************************************"
