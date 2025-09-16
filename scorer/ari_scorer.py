import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize
import re

@scorer_library.register(name="automated_readability_index")
class ARI(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.automated_readability_index(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def automated_readability_index(self, input: str) -> float:
        character_filter = re.compile(r'\S')
        num_characters = len(character_filter.findall(input))

        words = [word for word in word_tokenize(input) if word.isalpha()]
        sentences = sent_tokenize(input)

        score = 4.71 * (num_characters / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43

        return score
