import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@scorer_library.register(name="sentiment_consistency")
class SentimentConsistency(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.sentiment_consistency(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def sentiment_consistency(self, input: str) -> float:
        sentences = sent_tokenize(input)

        sid_obj = SentimentIntensityAnalyzer()
        sentiments = [sid_obj.polarity_scores(sentence)['compound'] for sentence in sentences]

        score = max(sentiments) - min(sentiments)
        return score