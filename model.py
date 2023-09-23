import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'Mensagem': ['Texto de alguma coisa', '0203', 'Eixo Interdisciplina:', 'Eixo Interdisciplinar:', 'Interdisciplinar:', 'Item do programa:', 'Item do programa 01:',  'Subitem do programa:'],
    'Classe': ['garbage', 'garbage', 'eixo', 'eixo', 'eixo', 'item', 'item', 'subitem']
}

df = pd.DataFrame(data)
print(df)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['Mensagem'])

model = MultinomialNB()

model.fit(X, df['Classe'])

nova_mensagem = ['eixooo']

nova_mensagem_vetorizada = vectorizer.transform(nova_mensagem)

resultado = model.predict(nova_mensagem_vetorizada)

print(resultado)  