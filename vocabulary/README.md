# vocabulary
*Script* para auxílio à produção do vocabulário para preparação de recursos e posterior treinamento de modelos de tópicos. Com esse *script*, é possível identificar as faixas de *tokens* de maior (menor) frequência em documentos, para remoção das mesmas caso válido. O *script* gera um gráfico mostrando as faixas de frequência candidatas à remoção.

### Executando
Para executar o *script* principal deste diretório, use ```python vocab_evaluation.py``` com os seguintes argumentos:

* ```dataset``` (*obrigatório*) - caminho do *dataset* pré-processado a ser analisado;
* ```dataset_name``` (*obrigatório*) - nome do *dataset* explorado;
* ```lang``` (*obrigatório*) - idioma do *dataset* sendo analisado. Pode receber ```"en"``` ou ```"pt"```;
* ```min_df_to_analyse``` (*obrigatório*) - faixa mínima de frequência em documentos (FD) para *tokens* a ser exibida;
* ```max_df_to_analyse``` (*obrigatório*) - faixa máxima de frequência em documentos (FD) para *tokens* a ser exibida;

Um exemplo de comando é o seguinte:

```shell
python vocab_evaluation.py \
    --dataset datasets/reddit_pt_2005_2020_desabafos_brasil[processed].json \
    --dataset_name 2005_2020_desabafos_brasil_pt \
    --lang pt \
    --min_df_to_analyse 0.1 \ # Primeira faixa de frequencia (10%) a ser considerada para exibicao no grafico gerado
    --max_df_to_analyse 1.0   # A contabilizacao de frequencias se dara ate essa faixa (100%)
```

### Links

* [What is Document Frequency (DF)?](https://kavita-ganesan.com/what-is-document-frequency/) - artigo descrevendo o conceito de frequência em documentos (DF) utilizado para análise e produção do vocabulário
