import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import syllables

@scorer_library.register(name="SMOG")
class SMOG(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.smog(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def smog(self, input: str) -> float:
        sentences = sent_tokenize(input)
        words = word_tokenize(input)
        
        num_polysyllabic_words = sum(1 for word in words if word.isalpha() and syllables.estimate(word) >= 3)    

        score = 1.0430 * math.sqrt(num_polysyllabic_words * (30 / len(sentences))) + 3.1291

        return score

