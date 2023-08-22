# %%
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("analise_nlp") \
    .getOrCreate()

#%%
dados = spark.read.csv("content/imdb-reviews-pt-br.csv",
                       escape="\"",
                       header=True,
                       inferSchema=True)

# Informações
print(f'Quantidade de linhas: {dados.count()} | Quantidade de colunas: {len(dados.columns)}')

dados.printSchema()

dados.limit(10).show()