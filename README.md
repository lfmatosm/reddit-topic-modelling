# reddit-topic-modelling
*Scripts* e utilitários para modelagem e identificação de tópicos relativos a depressão no *Reddit*, em língua portuguesa e inglesa, usando técnicas de modelagem de tópicos. Os modelos de tópicos *Latent Dirichlet Allocation* (LDA), *Contextualized Topic Model* (CTM) e *Embedded Topic Model* (ETM) foram explorados neste estudo. 

### :earth_americas: Ambiente
Use uma aplicação CLI compatível com *shell script* para executar os *scripts* de automatização de tarefas. A execução dos mesmos só foi realizada em ambiente Unix.

* [python >= 3.7.6](https://www.python.org/downloads/) - versão recomendada do python. A versão ```3.7.6``` da linguagem foi usada no projeto;
* [pip >= 20.x.x](https://pip.pypa.io/en/stable/installing/) - a versão ```20.2.4``` foi usada no projeto.

### :hammer: Configuração
Instale as dependências do projeto usando ```pip install -r requirements.txt```.

### :file_folder: Diretórios
Alguns diretórios nesse projeto representam funcionalidades usadas para realização da modelagem de tópicos. Essas pastas são descritas a seguir, e estão listadas de acordo com sua ordem recomendada de uso pra treinamento: 

* [preprocessing](./preprocessing) - possui o *script* para pré-processamento dos *corpora* explorados neste estudo, descrito por ```preprocess.py```. Esse *script* pode receber como entrada um *corpus* JSON contendo diversos documentos, onde cada um deles é representado por seu texto armazenado em seu campo ```"body"```. O *script* realiza tokenização e possibilita que etapas como lematização, remoção de *stopwords* e de categorias de *part-of-speech* (POS) indesejadas sejam removidas, tanto em português quanto em inglês.

* [vocabulary](./vocabulary) - possui o *script* ```vocab_evaluation.py```, usado para auxiliar a etapa de filtragem dos termos muito/pouco frequentes em um *corpus*. Após o pré-processamento do *corpus* realizado na etapa anterior, ainda podem restar no conjunto textual palavras que ocorrem com frequência alta ou baixa, e que não são interessantes para a modelagem. Esse *script* possibilita a geração de um histograma de faixas de frequência de palavras, que podem ser filtradas caso assim desejado pelo usuário.

* [preparation](./preparation) - possui o *script* para preparação de recursos para treinamento dos modelos de tópicos, descrito por ```prepare_training_resources.py```. Esse *script* prepara as entradas adequadas e recursos compartilhados para cada um dos três tipos de arquitetura de modelagem de tópicos aqui estudados: LDA, CTM e ETM. Note que o uso de *embeddings* de palavras SBERT e *word2vec* é necessário nesta etapa. O *corpus* pré-processado e o vocabulário/dicionário construídos na etapa anterior devem ser usados neste passo. O *script* produzirá dois conjuntos textuais, sendo o primeiro destinado ao treinamento dos modelos e o segundo destinado à sua validação.

* [training](./training) - possui os *scripts* para treinamento dos modelos de tópicos: ```lda.py```, ```ctm.py``` e ```etm.py```. A partir do *corpus* pré-processado e das entradas preparadas, cada treinamento é realizado. A métrica de coerência NPMI é calculada com base no conjunto textual de validação criado na etapa anterior.

### :floppy_disk: *Datasets*
Os *datasets* foram construídos por meio da [API Pushshift](https://github.com/pushshift/api), agregando postagens em português e em inglês relacionadas à discussões de depressão. Termos de busca tiveram de ser usados para coleta dos dados em português, dada a não existência de um *subreddit* voltado ao tema no idioma. A tabela a seguir detalha os *datasets* construídos. Ambos *corpora* estão disponíveis para uso na plataforma [Kaggle](https://www.kaggle.com/), em formato JSON.

| idioma          | subreddits usados | palavras-chave usadas                                                      |  período de coleta |  total de submissões coletadas | link |
| :-------------: |:----------------: | :------------------------------------------------------------------------: | :----------------: | :----------------------------: | :--: |
| português       | [brasil](https://www.reddit.com/r/brasil/), [desabafos](https://www.reddit.com/r/desabafos/) |   "depressão", "suicídio", "diagnóstico depressão", "tratamento depressão" |  2008-2021         | 3404                           | [1](https://www.kaggle.com/luizfmatos/reddit-portuguese-depression-related-submissions) |
| inglês          | [depression](https://www.reddit.com/r/depression/)        |   -                                                                        |  2009-2021         | 32165                          | [2](https://www.kaggle.com/luizfmatos/reddit-english-depression-related-submissions) |

### :pushpin: Licença
[MIT](LICENSE)
