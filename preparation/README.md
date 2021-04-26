# preparation
*Script* para preparar os recursos de treinamento dos modelos de tópicos, como conjuntos de treinamento e validação, entre outros. Recebe um *corpus* pré-processado, um dicionário produzido com o vocabulário do conjunto textual, entre outras informações necessárias para a preparação.

### Executando
Para executar o *script* principal deste diretório, use ```python prepare_training_resources.py```. Os argumentos que o script recebe são os seguintes:

* ```dataset``` (*obrigatório*) - caminho do *dataset* pré-processado;
* ```dictionary``` (*obrigatório*) - caminho do dicionário associado ao *dataset*;
* ```embeddings``` (*obrigatório*) - caminho dos *embeddings* *word2vec* a serem preparados para o treinamento de modelos ETM;
* ```lang``` (*obrigatório*) - idioma do *dataset* sendo preparado. Pode receber ```"en"``` ou ```"pt"```;
* ```n_dim``` - número de dimensões dos *embeddings*. Por padrão tem o valor ```300```;
* ```train_size``` - fração de documentos a serem reservados para treino. O restante será reservado para validação. Deve receber uma fração; por padrão tem o valor ```1.0```, reservando todos os documentos para treinamento;
* ```dataset_name``` - nome do *dataset* sendo usado. É empregado para gerar o diretório onde serão armazenados os recursos de treinamento, que pode ser encontrado em ```resources/<dataset_name>```. Por padrão tem o valor ```training_data```.

Um exemplo de comando é o seguinte:

```shell
python prepare_training_resources.py \
    --dataset datasets/reddit_pt_2005_2020_desabafos_brasil[processed].json \
    --dictionary example/dicionary.gdict \
    --embeddings embeddings/ptwiki_20180420_300d_optimized.w2v \
    --lang pt \
    --n_dim 300 \
    --train_size 0.8 \
    --dataset_name 2005_2020_desabafos_brasil_pt
```
