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

#%%

#Limpeza dos dados
''' 
Caracteres Especiais
Espaços antes e depois dos textos
stopwords (preposições, pronomes, verbos)
normalização (Variação de flexão, número e grau) Ex. Amigos, amgas, amigo para amig.

'''
# wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dados.limit(10).show(truncate=False)

amostra = dados.select('text_pt').sample(fraction =  0.1, seed = 101)
tudo = [texto['text_pt'] for texto in amostra.collect()] 

#%%
wordCloud = WordCloud(
    collocations=False,
    background_color='white',
    prefer_horizontal=1,
    width=1000,
    height=600
).generate(str(tudo))

#%%
plt.figure(figsize=(20,8))
plt.axis('off')
plt.imshow(wordCloud)


# %%
# removendo caracteres especiais

import string 
import pyspark.sql.functions as f
string.punctuation

df = spark.createDataFrame([("oi # sd ! % [()] > ?",),("@ asd ",)],["textos"])

dados = dados.withColumn("text_regex", f.regexp_replace("text_en", r"[^\w\s]", ""))
dados = dados.withColumn("text_trim", f.trim(dados.text_regex))
dados.limit(10).show(truncate=False)

#%%
#Tokenização
from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="text_trim", outputCol="tokens")
tokenizado = tokenizer.transform(dados)

#%%
from pyspark.sql.types import IntegerType
#tokenizado.select("text_trim","tokens").show()
countTokens = f.udf(lambda tokens: len(tokens), IntegerType())
tokenizado.select("text_trim","tokens").withColumn("freq_token", countTokens(f.col("tokens"))).show(truncate=False)

#%%

#Stopwords

data = [(0, "Spark é ótimo com NLP"),
        (1, "Spark MLlib não ajuda"),
        (2, "O MLlib do spark ajuda e é facil")]
columns = ['label', 'texto_limpo']

df = spark.createDataFrame(data, columns)
# NLTK
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
stop_A = stopwords.words("portuguese")

#%%
#pySpark
from pyspark.ml.feature import StopWordsRemover

stop_B = StopWordsRemover.loadDefaultStopWords("portuguese")

stop_B

#%%

tokenizer2 = Tokenizer(inputCol="texto_limpo", outputCol="tokens")
tokenizado2 = tokenizer2.transform(df)

#%%
remover = StopWordsRemover(inputCol="tokens", outputCol="texto_final", stopWords=stop_A)
df = remover.transform(tokenizado2)

#%%
df.show()

#%%

remover = StopWordsRemover(inputCol="tokens", outputCol="texto_final", stopWords=stop_B)
df = remover.transform(tokenizado2)

df.show()

#%%

tokenizer3 = Tokenizer(inputCol="text_trim", outputCol="tokens")
tokenizado3 = tokenizer3.transform(dados)

remover = StopWordsRemover(inputCol="tokens", outputCol="text_final", stopWords=stop_B)
df = remover.transform(tokenizado3)

#%% 
df.show(truncate=False)