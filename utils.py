# Retorna um Dataframe com os metadados das notícias do grupo selecionado
def import_metadata(tipo):
    import pandas as pd
    import os
    
    noticias_meta_header = ["author", "link", "category", "date_of_publication", "number_of_tokens", "words_without_punct", "number_of_types",
                            "number_of_links", "upper_case_words", "number_of_verbs", "subj_and_imp_verbs", "number_of_nouns", "number_of_adjectives",
                            "number_of_adverbs", "number_modal_verbs", "sing_first_sec_personal_pronouns", "plural_first_personal_pronouns", "number_of_pronouns",
                            "pausality", "number_of_characters", "avg_sentence_length", "average_word_length", "perc_news_with_speeling_errors", 
                            "emotiveness", "diversity", "id"]

    noticias_meta = pd.DataFrame(columns=noticias_meta_header)

    caminho_dados = ("full_texts/fake-meta-information/" if tipo == "fake" else "full_texts/true-meta-information/")

    for noticia in sorted(os.listdir(caminho_dados)):
        with open(caminho_dados + noticia, 'r') as f:
            if tipo == "fake":
                data = f.read().replace('\n',',')
                separator = ","
            else:
                data = f.read().replace('\n',';')
                separator = ";"
            nova_noticia = pd.read_csv(pd.compat.StringIO(data), sep = separator, names = noticias_meta_header)
            nova_noticia["id"] = noticia
            noticias_meta = noticias_meta.append(nova_noticia, ignore_index=True)

    return noticias_meta

# Retorna um dicionário com os textos das notícias selecionadas
def import_texto(tipo):
    import pandas as pd
    import os
    from nltk import word_tokenize
    
    caminho_dados = ("size_normalized_texts/fake/" if tipo == "fake" else "size_normalized_texts/true/")
    
    full_texts = {}
    
    for noticia in sorted(os.listdir(caminho_dados)):
        with open(caminho_dados + noticia, 'r') as f:
            data = list(map(lambda x : x.lower(), word_tokenize(f.read())))
            full_texts[noticia] = data
            
    return full_texts

def normaliza_texto(noticias):
    import nltk
    import re
      
    stopwords = set(nltk.corpus.stopwords.words('portuguese')) # Usar stopwords como lista é lento para grandes documentos
    pontuacao = set([",", ".", "'", "\"", "“", "”", "(", ")", "{", "}", "[", "]", "!", "?", "^", "~", "-", "`", "'", ":", ">", "<", "+", "-", "_", "''", "``", "...", "–",
                    "%", "$", "&"])
    
    stemmer = nltk.stem.RSLPStemmer()
    
    for texto in sorted(noticias.keys()):
        corpo = noticias[texto]
        
        # Removendo stopwords e pontuação
        corpo = list(filter(lambda x : (x not in stopwords) and
                            (x not in pontuacao) and
                            (not re.search("^[0-9]*$", x)) and
                            (not re.search("([0-1][0-9]|2[0-3]):[0-5][0-9]", x)), corpo))
        noticias[texto] = list(map(lambda x : stemmer.stem(x), corpo))
        
    return noticias