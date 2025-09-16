import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize
import syllables

@scorer_library.register(name="flesh_kincaid_grade_level")
class FleshKincaid(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.flesch_Kincaid_grade_level(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def flesch_Kincaid_grade_level(self, input: str) -> float:
        words = words = [word for word in word_tokenize(input) if word.isalpha()]
        sentences = sent_tokenize(input)

        num_syllables = sum(syllables.estimate(word) for word in words)

        score = 0.39 * (len(words) / len(sentences)) + 11.8 * (num_syllables / len(words)) - 15.59

        return score