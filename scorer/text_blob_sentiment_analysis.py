import weave
from scorer.library import scorer_library
from  textblob import TextBlob

@scorer_library.register(name="text_blob_sentiment_analysis")
class TextBlobSentimentAnalysis(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.text_blob_sentiment_analysis(content["text"])
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def text_blob_sentiment_analysis(self, input: str) -> float:
        blob = TextBlob(input)

        # Polarity: A float between -1 (negative) and 1 (positive), representing the sentiment polarity.
        # Subjectivity: A float between 0 (objective) and 1 (subjective), representing how subjective or opinionated the text is.

        scores = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }

        return scores