# reddit-topic-modelling
*Scripts*, *datasets* e utilitários para modelagem e identificação de tópicos relativos a depressão no *Reddit* usando *Latent Dirichlet Allocation* (LDA).

### Configuração
Ative o ambiente Pipenv na raiz do projeto usando ```pipenv shell```

A seguir, para instalar todas as dependências dos projetos-filhos, execute ```pipenv install```

Agora você provavelmente poderá executar quaisquer dos projetos-filhos sem problemas. Se quiser instalar apenas um dos projetos, entre na pasta referente e siga as instruções do mesmo.

### Projetos
Os utilitários/*scripts* presentes neste projeto são, de forma geral, independentes do *dataset* utilizado - com exceção do *script* ```posts-gatherer```. Portanto, seu código pode ser facilmente reaproveitado para modelagem LDA em outros contextos. A seguir, há uma breve descsrição sobre o que cada projeto faz:

* ```lda-trainer``` - *scripts* de treinamento LDA. Deve ser usado apenas quando possuir um *dataset* pré-processado e em seu formato de entrada (conjunto de registros em um arquivo JSON)
* ```models-evaluation``` - *notebook* Jupyter e scripts para avaliação de resultados de treinamento
* ```posts-gatherer``` - *script* para coleta automatizada de postagens do *Reddit*. Você determina em que banco Mongo os dados devem ser salvos e *keywords*/*subreddits* para serem buscados, e o *script* executa a tarefa. Usado para montagem do *dataset* original para treinamento
* ```text-preprocessor``` - *scripts* para pré-processamento de um *dataset*. Remoção de *stopwords*, *tokenização* e *lemmatização* são realizadas aqui. Um *dataset* deve ser pré-processado antes de ser submetido a treinamento no ```lda-trainer```

### *Datasets*
Os *datasets*, original e pré-processado utilizados no presente trabalho encontram-se na pasta de mesmo nome. Observe que as bases de dados são arquivos no formato JSON, onde cada registro é um objeto representando uma submissão dentro do *Reddit*.

### Executando
Cada projeto-filho possui sua forma de uso, detalhada em seus README. De forma geral os *scripts* são executáveis por linha de comando.

### Resultados
Uma avaliação dos resultados de treinamento obtidos no presente trabalho é realizada [neste notebook](https://github.com/lffloyd/reddit-topic-modelling/blob/master/models-evaluation/Reddit_pt%20-%20Modelagem%20de%20t%C3%B3picos%20-%20Resultados%20de%20treinamento.ipynb).

### Licença
[MIT](LICENSE)