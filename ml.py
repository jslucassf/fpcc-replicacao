from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pandas as pd

def transform_bow(data):
    for i in data:
        data.loc[data[i] > 0, i] = 1
    return data.drop('non_immediacy', axis = 1)

# Features possíveis:
#     pos - POS tags
#     bow - Bag of Words
#     pos+bow
#     pau - Pausality
#     emo - Emotiveness
#     unc - Uncertainty
#     nim - Non-Immediacy
#     p+e+u+n - Pausality + emotiveness + Uncertainty + Non-Immediacy
#     all - Todas
def get_features(dataframe_base, features):
    if(features == "pos"):
        return dataframe_base[["number_of_verbs", "subj_and_imp_verbs", "number_of_nouns", "number_of_adjectives", "number_of_adverbs", "number_modal_verbs",
                               "number_of_pronouns"]]
    elif(features == "bow"):
        return transform_bow(dataframe_base.iloc[:, 26:])
    elif(features == "pos+bow"):
        noticias_pos = dataframe_base[["number_of_verbs", "subj_and_imp_verbs", "number_of_nouns", "number_of_adjectives", "number_of_adverbs", "number_modal_verbs",
                               "number_of_pronouns"]]
        noticias_bow = transform_bow(dataframe_base.iloc[:, 26:])
        return pd.concat([noticias_pos, noticias_bow], axis=1, join_axes=[noticias_pos.index])
    elif(features == "pau"):
        return dataframe_base[["pausality"]]
    elif(features == "emo"):
        return dataframe_base[["emotiveness"]]
    elif(features == "unc"):
        return dataframe_base[["number_modal_verbs"]]
    elif(features == "nim"):
        return dataframe_base[["non_immediacy"]]
    elif(features == "p+e+u+n"):
        return dataframe_base[["pausality", "emotiveness", "number_modal_verbs", "non_immediacy"]]
    elif(features == "bow+e"):
        noticias_bow_emot = transform_bow(dataframe_base.iloc[:, 26:])
        return pd.concat([noticias_bow_emot, dataframe_base[["emotiveness"]]], axis = 1, join_axes = [noticias_bow_emot.index])
    elif(features == "all"):
        return dataframe_base.loc[:, dataframe_base.columns != 'true']

def train_evaluate(dataframe_base, features = "all"):
    X = get_features(dataframe_base, features)
    y = dataframe_base.true
    
    clf = svm.SVC(kernel = "linear")
    pred = cross_val_predict(clf, X, y, cv = 5)
    
    del X, clf
    
    return classification_report(y, pred)