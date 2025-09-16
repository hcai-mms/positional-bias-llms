import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize
import syllables

@scorer_library.register(name="flesh_reading_ease")
class FleshReadingScorer(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.flesch_reading_ease(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def flesch_reading_ease(self, input: str) -> float:
        words = words = [word for word in word_tokenize(input) if word.isalpha()]
        sentences = sent_tokenize(input)

        num_syllables = sum(syllables.estimate(word) for word in words)

        score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (num_syllables / len(words))

        return score