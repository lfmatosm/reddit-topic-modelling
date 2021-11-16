### Execução

```
python octis_test/unified_training.py \
    --dataset_path <DATASET_PATH> \
    --embeddings_path <EMBEDDINGS_PATH> \
    --models <LIST OF "etm", "lda", "ctm"> \
    --all-models <BOOLEAN>
```

python octis_test/unified_training.py --models etm --dataset_path /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/octis_test/datasets/tsv/pt --embeddings_path /home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/embeddings/ptwiki_20180420_300d_optimized.w2v