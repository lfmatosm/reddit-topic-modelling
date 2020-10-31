#!/bin/bash

train_batch_name=$1

echo "Treinamento: $train_batch_name"

echo "Gerando datasets de treino..."
python3 scripts/split_dataset_for_training.py \
    --dataset /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/datasets/processed/reddit_gatherer.pt_submissions[original_dataset][2008_2020][nouns].json
echo "Datasets de treino gerados"

echo "Iniciando treinamento de modelos ctm, lda e etm..."
# pipenv shell

declare -a models=("ctm" "lda" "etm")

pids=()

# Treinamento CTM
# python3 top2vec-and-ctm/ctm_training.py \
#     --dataset datasets_for_training/ctm_dataset.txt \
#     --topics 5 8 10 12 15 18 20 22 25 28 30 &
python3 top2vec-and-ctm/ctm_training.py \
    --dataset datasets_for_training/ctm_dataset.txt \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --topics 5 &
#Salva PID de cada tarefa iniciada para posterior verificacao de status de finalizacao
pids+=($!)

# Treinamento LDA
# python3 lda-trainer/train_lda.py \
#     --dataset /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/datasets_for_training/training_dataset.json \
#     --topics 5 8 10 12 15 18 20 22 25 28 30 &
python3 lda-trainer/train_lda.py \
    --dataset /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/datasets_for_training/training_dataset.json \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --topics 5 &
pids+=($!)

# Treinamento ETM
# python etm-trainer/main.py \
#     --mode train --dataset reddit_pt_nouns \
#     --data_path datasets_for_training/min_df_0.01  \
#     --original_data_path datasets_for_training/training_dataset.json \
#     --emb_path etm-trainer/skip_s300.txt \
#     --topics 5 8 10 12 15 18 20 22 25 28 30 \
#     --train_embeddings 0 --epochs 1000 &
python etm-trainer/main.py \
    --mode train --dataset reddit_pt_nouns \
    --data_path datasets_for_training/min_df_0.01  \
    --original_data_path datasets_for_training/training_dataset.json \
    --dictionary datasets_for_training/word_dictionary.gdict \
    --emb_path etm-trainer/skip_s300.txt \
    --topics 5 \
    --train_embeddings 0 --epochs 1 &
pids+=($!)


errors=()

# Aguarda finalizacao das tarefas iniciadas. Caso alguma falhe,
# insere o nome do projeto que falhou no array de erros
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        errors+=(${models[$i]})
    fi
done

# Caso tenha ocorrido algum erro no treinamento, exibe os que falharam 
# com problemas
if (( ${#errors[@]} )); then
    echo "ERRO - Treinamento dos seguintes modelos falhou: ${errors[@]}"
    exit 1
fi

mkdir -p models_evaluation/models_training/$train_batch_name
cp -R models_training models_evaluation/models_training/$train_batch_name
rm -rf models_training
rm -rf results
echo "Resultados de treinamento armazenados em 'models_evaluation/models_training/$train_batch_name'"

echo "Treinamento de modelos ctm, lda e etm realizado com sucesso"
