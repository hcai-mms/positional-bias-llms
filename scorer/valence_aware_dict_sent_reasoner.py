import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@scorer_library.register(name="valence_aware_dict_sent_reasoner")
class ValeneceAwareDictSentReasoner(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.valence_aware_dictionary_and_sentiment_reasoner(content["text"])
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def valence_aware_dictionary_and_sentiment_reasoner(self, input: str) -> float:
        sid_obj = SentimentIntensityAnalyzer()

        # pos: Positive sentiment score
        # neu: Neutral sentiment score
        # neg: Negative sentiment score
        # compound: A normalized, weighted composite score of the overall sentiment (-1 to +1))
        sentiment_dict = sid_obj.polarity_scores(input)

        #? Maybe only return compound value 
        return sentiment_dict

