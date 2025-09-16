import weave
from typing import Any
from collections import defaultdict
from scorer.library import scorer_library
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.library import model_library
import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('wordnet')


stemmer = nltk.stem.porter.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer ()

tokenizer_reg = RegexpTokenizer(r'\w+')

def normalize_stemming(text):
    tokens = tokenizer_reg.tokenize(text)
    return [stemmer.stem(item) for item in tokens]

def normalize_lemma(text):
    tokens = tokenizer_reg.tokenize(text)
    return [lemmatizer.lemmatize(item) for item in tokens]


@scorer_library.register(name="golden_description_cosine_tfidf")
class GoldenDescriptionCosineTFIDFScorer(weave.Scorer):
    golden_descritions:dict

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._golden_descritpion_vectorizer_stem = {k:TfidfVectorizer(tokenizer=normalize_stemming, stop_words='english') for k in self.golden_descritions.keys()}
        self._golden_descritpion_vectorizer_lemma = {k:TfidfVectorizer(tokenizer=normalize_lemma, stop_words='english') for k in self.golden_descritions.keys()}

        self._golden_descritpion_tfidf_stem = {k:self._golden_descritpion_vectorizer_stem[k].fit_transform(descriptions) for k, descriptions in self.golden_descritions.items()}
        self._golden_descritpion_tfidf_lemma = {k:self._golden_descritpion_vectorizer_lemma[k].fit_transform(descriptions)for k, descriptions in self.golden_descritions.items()}


    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.golden_cosine(content["text"], topic)
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def golden_cosine(self, input: str, topic) -> float:
        input_tfidf_stem = self._golden_descritpion_vectorizer_stem[topic].transform([input])
        input_tfidf_lemma = self._golden_descritpion_vectorizer_lemma[topic].transform([input])

        return {
            "stemmed": np.mean(cosine_similarity(input_tfidf_stem, self._golden_descritpion_tfidf_stem[topic])).item(),
            "lemmatized": np.mean(cosine_similarity(input_tfidf_lemma, self._golden_descritpion_tfidf_lemma[topic])).item(),
        }
