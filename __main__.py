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

#%%

print("Negativo")
dados.filter(dados.id == 190).select("text_pt").show(truncate=False)

print("Positivo")
dados.filter(dados.id == 12427).select("text_pt").show(truncate=False)

#%%

# Contabilizano os tipos de dados
dados.groupBy('sentiment').count().show()