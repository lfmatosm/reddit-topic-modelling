#!/bin/bash

train_batch_name=$1
preprocessed_dataset=$2
embedding=$3

echo "*************************************************************************************"

echo "Treinamento: $train_batch_name"

echo "Gerando datasets de treino..."
python3 scripts/split_dataset_for_training.py \
    --dataset $preprocessed_dataset
echo "Datasets de treino gerados"

echo "Iniciando treinamento de modelos ctm, lda e etm..."


# Treinamento CTM
python3 ctm/ctm_training.py \
    --dataset datasets_for_training/ctm_dataset.txt \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --topics 5 8 10 12 15 18 20 22 25 28 30

# Treinamento LDA
python3 lda-trainer/train_lda.py \
    --dataset /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/datasets_for_training/training_dataset.json \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --topics 5 8 10 12 15 18 20 22 25 28 30

# Treinamento ETM
python etm-trainer/main.py \
    --mode train --dataset $train_batch_name \
    --data_path datasets_for_training/min_df_0.01  \
    --original_data_path datasets_for_training/training_dataset.json \
    --emb_path $embedding \
    --topics 5 8 10 12 15 18 20 22 25 28 30 \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --train_embeddings 0 --epochs 1000

mkdir -p models_evaluation/models_training/$train_batch_name
cp -R models_training models_evaluation/models_training/$train_batch_name
cp -R datasets_for_training models_evaluation/models_training/$train_batch_name
rm -rf models_training
rm -rf datasets_for_training
rm -rf results
echo "Resultados de treinamento armazenados em 'models_evaluation/models_training/$train_batch_name'"

echo "Treinamento de modelos ctm, lda e etm realizado com sucesso"

echo "*************************************************************************************"
