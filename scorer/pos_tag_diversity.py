import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('averaged_perceptron_tagger_eng')

@scorer_library.register(name="pos_tag_diversity")
class PosTagDiversity(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.pos_tag_diversity(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def pos_tag_diversity(self, input: str) -> float:
        words = word_tokenize(input)
        tags = [tag for _, tag in pos_tag(words)]

        score = len(set(tags)) / len(tags)
        return score