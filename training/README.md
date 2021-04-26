# training
*Scripts* para treinamento dos modelos LDA, CTM e ETM. Utiliza os recursos preparados com o *script* de preparação para realização dos treinamentos.

### Executando
Existem três *scripts* no diretório, sendo cada um destinado a um tipo de treinamento. Os *scripts* neste diretório geram os modelos finais no diretório ```training_outputs/models/<lda|ctm|etm>``` e os resultados de treinamento são agregados em tabelas CSV que podem ser encontradas em ```training_outputs/csvs/<lda|ctm|etm>``` A seguir, as instruções de uso para cada um deles são descritas.

#### Executando [lda.py](./lda.py)
O *script* executa o treinamento de modelos LDA por meio da [implementação do pacote scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html). Para executar o *script* , use ```python lda.py``` passando os argumentos a seguir:

* ```dataset_name``` (*obrigatório*) - nome do *dataset* sendo usado para treinamento;
* ```lang``` (*obrigatório*) - idioma do *dataset*. Pode ser ```"en"``` ou ```"pt"```;
* ```train_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para treinamento. Esse arquivo é gerado pelo *script* de preparação;
* ```validation_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para validação. O cálculo da métrica de coerência NPMI será realizado com este conjunto de dados. Esse arquivo é gerado pelo *script* de preparação;
* ```dictionary``` (*obrigatório*) - caminho do dicionário do vocabulário do *corpus*, gerado pelo *script* de produção de vocabulário;
* ```topics``` (*obrigatório*) - lista de inteiros com os valores do hiperparâmetro K a serem usados para treinamento dos modelos. Um modelo será treinado para cada valor passado nessa lista;

Um exemplo de comando é o seguinte:

```shell
python lda.py \
    --dataset_name 2005_2020_desabafos_brasil_pt \
    --lang pt \
    --train_documents resources/2005_2020_desabafos_brasil_pt/train_documents.json  \
    --validation_documents resources/2005_2020_desabafos_brasil_pt/validation_documents.json  \
    --dictionary resources/2005_2020_desabafos_brasil_pt/dictionary.gdict \
    --topics 5 7 10 12 15 18 20
```

#### Executando [ctm.py](./ctm.py)
O *script* executa o treinamento de modelos CTM por meio da [implementação do pacote contextualized_topic_models](https://github.com/MilaNLProc/contextualized-topic-models#readme). Para executar o *script* , use ```python ctm.py``` passando os argumentos a seguir:

* ```dataset_name``` (*obrigatório*) - nome do *dataset* sendo usado para treinamento;
* ```lang``` (*obrigatório*) - idioma do *dataset*. Pode ser ```"en"``` ou ```"pt"```;
* ```train_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para treinamento. Esse arquivo é gerado pelo *script* de preparação;
* ```validation_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para validação. O cálculo da métrica de coerência NPMI será realizado com este conjunto de dados. Esse arquivo é gerado pelo *script* de preparação;
* ```dictionary``` (*obrigatório*) - caminho do dicionário do vocabulário do *corpus*, gerado pelo *script* de produção de vocabulário;
* ```topics``` (*obrigatório*) - lista de inteiros com os valores do hiperparâmetro K a serem usados para treinamento dos modelos. Um modelo será treinado para cada valor passado nessa lista;
* ```data_preparation``` (*obrigatório*) - caminho para o objeto de preparação do CTM, gerado pelo *script* de preparação;
* ```prepared_training_dataset``` (*obrigatório*) - caminho para o conjunto de treinamento preparado do CTM, gerado pelo *script* de preparação;
* ```inference``` - tipo de inferência a ser usado pelo CTM. Por padrão, usa ```"combined"```;

Um exemplo de comando é o seguinte:

```shell
python ctm.py \
    --dataset_name 2005_2020_desabafos_brasil_pt \
    --lang pt \
    --train_documents resources/2005_2020_desabafos_brasil_pt/train_documents.json  \
    --validation_documents resources/2005_2020_desabafos_brasil_pt/validation_documents.json  \
    --dictionary resources/2005_2020_desabafos_brasil_pt/dictionary.gdict \
    --topics 5 7 10 12 15 18 20 \
    --data_preparation $base_prepared_resources_dir/ctm_data_preparation.obj \
    --prepared_training_dataset $base_prepared_resources_dir/ctm_training_dataset.dataset
```

#### Executando [etm.py](./etm.py)
O *script* executa o treinamento de modelos ETM por meio da [implementação do pacote embedded-topic-model](https://github.com/lffloyd/embedded-topic-model#readme). Para executar o *script* , use ```python etm.py``` passando os argumentos a seguir:

* ```dataset_name``` (*obrigatório*) - nome do *dataset* sendo usado para treinamento;
* ```lang``` (*obrigatório*) - idioma do *dataset*. Pode ser ```"en"``` ou ```"pt"```;
* ```train_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para treinamento. Esse arquivo é gerado pelo *script* de preparação;
* ```validation_documents``` (*obrigatório*) - caminho do arquivo do conjunto de documentos reservados para validação. O cálculo da métrica de coerência NPMI será realizado com este conjunto de dados. Esse arquivo é gerado pelo *script* de preparação;
* ```dictionary``` (*obrigatório*) - caminho do dicionário do vocabulário do *corpus*, gerado pelo *script* de produção de vocabulário;
* ```topics``` (*obrigatório*) - lista de inteiros com os valores do hiperparâmetro K a serem usados para treinamento dos modelos. Um modelo será treinado para cada valor passado nessa lista;
* ```training_dataset``` (*obrigatório*) - caminho para o arquivo de documentos de treinamento específico do ETM, gerado pelo *script* de preparação;
* ```embeddings``` (*obrigatório*) - caminho para o arquivo de *embeddings* *word2vec* a ser usado;
* ```vocabulary``` (*obrigatório*) - caminho para arquivo de vocabulário para ETM, gerado pelo *script* de preparação;

Um exemplo de comando é o seguinte:

```shell
python etm.py \
    --dataset_name 2005_2020_desabafos_brasil_pt \
    --lang pt \
    --train_documents resources/2005_2020_desabafos_brasil_pt/train_documents.json  \
    --validation_documents resources/2005_2020_desabafos_brasil_pt/validation_documents.json  \
    --dictionary resources/2005_2020_desabafos_brasil_pt/dictionary.gdict \
    --topics 5 7 10 12 15 18 20 \
    --training_dataset resources/2005_2020_desabafos_brasil_pt/etm_training_dataset.dataset \
    --vocabulary resources/2005_2020_desabafos_brasil_pt/etm_vocabulary.vocab \
    --embeddings embeddings/ptwiki_20180420_300d_optimized.w2v
```
