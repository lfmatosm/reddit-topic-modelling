#!/bin/bash

train_batch_name=$1
preprocessed_dataset=$2
min_df=$3
max_df=$4

echo -e "\n*************************************************************************************"

echo -e "Training: $train_batch_name"

echo -e "\nPreparing training resources (datasets, vocabulary, dictionary, etc)..."
python scripts/prepare_dataset_for_training.py \
    --dataset $preprocessed_dataset --dataset_name $train_batch_name \
    --min_df $min_df --max_df $max_df
echo -e "\nTraining resources prepared"

base_prepared_resources_dir="resources/$train_batch_name"

echo -e "\nCopying original preprocessed dataset to '$base_prepared_resources_dir' folder..."
cp $preprocessed_dataset $base_prepared_resources_dir
echo -e "\nMoved dataset to '$base_prepared_resources_dir' folder"

echo -e "\nStarting ctm, lda and etm training..."

# CTM
echo -e "\nStarting ctm training...\n"
python training/ctm.py \
    --split_documents $base_prepared_resources_dir/split_documents.json \
    --data_preparation $base_prepared_resources_dir/ctm_data_preparation.obj \
    --prepared_training_dataset $base_prepared_resources_dir/ctm_training_dataset.dataset \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --topics 5 8 10 12 15 18 20 22 25 28 30

# LDA
echo -e "\nStarting lda training...\n"
python training/lda.py \
    --split_documents $base_prepared_resources_dir/split_documents.json \
    --joined_documents $base_prepared_resources_dir/joined_documents.json \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --topics 5 8 10 12 15 18 20 22 25 28 30


# ETM
echo -e "\nStarting etm training...\n"
python training/etm.py \
    --split_documents $base_prepared_resources_dir/split_documents.json \
    --training_dataset $base_prepared_resources_dir/etm_training_dataset.dataset \
    --vocabulary $base_prepared_resources_dir/etm_vocabulary.vocab \
    --dictionary $base_prepared_resources_dir/word_dictionary.gdict \
    --embeddings skip_s300.txt \
    --topics 5 8 10 12 15 18 20 22 25 28 30

echo -e "\nTraining finished successfully"

notebook_name="$(date '+%Y-%m-%d')_$train_batch_name"
notebook_extension=".ipynb"
evaluation_base_path="evaluation/$notebook_name"

echo -e "\nCreating evaluation folder at '$evaluation_base_path' and moving training outputs to the folder..."
mkdir -p $evaluation_base_path
cp -R training_outputs/csvs $evaluation_base_path
cp -R training_outputs/models $evaluation_base_path
cp -R resources $evaluation_base_path
rm -rf training_outputs
rm -rf resources

echo -e "\nCreating notebook at '$evaluation_base_path$notebook_extension'..."
cp -R evaluation_example/utils $evaluation_base_path
cp evaluation_example/evaluation_example.ipynb $evaluation_base_path/$notebook_name$notebook_extension

echo -e "\nCleaning generated files..."
rm -rf training_outputs
rm -rf resources
echo -e "\nFolder cleaned"

echo -e "\nPostprocessing finished successfully"

echo -e "\n*************************************************************************************"
