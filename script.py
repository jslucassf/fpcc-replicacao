import pandas as pd
import numpy as np
import os
import utils

noticias_fake = utils.import_metadata("fake")
noticias_fake["true"] = 0
noticias_true = utils.import_metadata("true")
noticias_true["true"] = 1

noticias = noticias_fake.append(noticias_true)
noticias = noticias.drop(["author", "link", "category", "date_of_publication", "number_of_tokens", "words_without_punct", "number_of_types",
               "number_of_links", "upper_case_words", "id"], axis = 1)

noticias_fake_corpo = utils.import_texto("fake")
noticias_true_corpo = utils.import_texto("true")

noticias_fake_corpo = utils.normaliza_texto(noticias_fake_corpo)
noticias_true_corpo = utils.normaliza_texto(noticias_true_corpo)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

corpus = []

for noticia in sorted(noticias_fake_corpo.keys()):
    corpus.append(" ".join(noticias_fake_corpo[noticia]))
for nocicia in sorted(noticias_true_corpo.keys()):
    corpus.append(" ".join(noticias_true_corpo[noticia]))

X = vectorizer.fit_transform(corpus)

bow = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
bow = bow.iloc[:,520:]

result = pd.concat([noticias, bow], axis=1, join_axes=[noticias.index])

# Liberando memória
del noticias_fake, noticias_true, noticias_fake_corpo, noticias_true_corpo, vectorizer, corpus, X, bow

# Modelo 1: POS tags
noticias_pos = result[["number_of_verbs", "subj_and_imp_verbs", "number_of_nouns", "number_of_adjectives", "number_of_adverbs", "number_modal_verbs",
                           "number_of_pronouns", "true"]]

# Modelo 2: Bag of Words
noticias_bow = result.iloc[:, 26:]

# Modelo 3: POS + BoW
noticias_pos_bow = pd.concat([noticias_pos, noticias_bow], axis=1, join_axes=[noticias_pos.index])
print(noticias_bow.head())
