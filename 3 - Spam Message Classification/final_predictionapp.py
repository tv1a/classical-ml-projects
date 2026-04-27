import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open("spam.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_model.pkl", "rb") as f:
    vectorizer = pickle.load(f)

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stopWords = stopwords.words("english")


def predictSpam(message):
    message = message.lower()
    message = re.sub(r"[^\w\s]", "", message)
    message = word_tokenize(message)
    message = " ".join(message)
    vectorized = vectorizer.transform([message])

    preds = model.predict_proba(vectorized)
    probSpam = preds[0][1]
    return "SPAM" if probSpam > 0.25 else "NOT SPAM"


while True:
    text = input()
    prediction = predictSpam(text)
    print(prediction)
