# preprocessing
*Script* para pré-processar um *corpus* simples de documentos armazenado em um arquivo JSON. O *script* realiza remoção de caracteres especiais, tokenização, remoção de *stopwords*, lematização e remoção de caregorias de part-of-speech (POS).

### Executando
Para executar o *script* principal deste diretório, use ```python preprocess.py```. Os argumentos que o script recebe são os seguintes:

* ```datasetFile``` (*obrigatório*) - caminho do arquivo do *dataset*. Deve ser um arquivo JSON;
* ```datasetName``` (*obrigatório*) - nome do *dataset* a ser pré-processado. Usado para gerar o nome do arquivo pós-processado;
* ```datasetFolder``` (*obrigatório*) - diretório onde o *dataset* será salvo após seu processamento. O diretório poderá ser encontrado no caminho ```datasets/processed/<datasetFolder>```;
* ```field``` (*obrigatório*) - campo a ser pré-processado do *dataset*. Deve ser um campo existente nos documentos presentes no *dataset* em formato JSON passado;
* ```lang``` (*obrigatório*) - idioma do *corpus*. Pode receber ```"en"``` ou ```"pt"```;
* ```lemmatize``` - indica se palavras devem ser reduzidas aos seus lemas ou não. Recebe ```True``` para ativar ou ```False``` para desativar. Por padrão, está desativado;
* ```removeStopwords``` - indica se deve remover *stopwords*. As *stopwords* serão procuradas com base em listas de palavras definidas para o idioma passado no argumento ```lang```. Caso ```stopwordsFile``` seja passado, as palavras presentes no arquivo também serão removidas caso encontradas no *corpus*. Recebe ```True``` para ativar ou ```False``` para desativar. Por padrão, está desativado;;
* ```stopwordsFile``` - arquivo com *stopwords* adicionais para remoção;
* ```removePos``` - indica se deve remover categorias de POS. As categorias de POS devem ser indicadas no argumento ```desiredPos```. Recebe ```True``` para ativar ou ```False``` para desativar. Por padrão, está desativado;;
* ```desiredPos``` - lista de categorias de POS que devem ser removidas do *corpus*. Caso seja passado, ```removePos``` deve ser passado com o valor ```True```. As categorias seguem [a nomenclatura usada pela biblioteca spaCy](https://spacy.io/usage/linguistic-features#pos-tagging).

Um exemplo de comando é o seguinte:

```python preprocess.py --datasetFile datasets/reddit_pt_2005_2020_desabafos_brasil.json --datasetName 2005_2020_desabafos_brasil_pt --datasetFolder example_datasets --field body --lang pt --lemmatize True --removePos True --desiredPos NOUN VERB ADJ --removeStopwords True```
