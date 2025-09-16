import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize

@scorer_library.register(name="type_token_ratio")
class TypeToken(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.type_token_ratio(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def type_token_ratio(self, input: str) -> float:
        words = word_tokenize(input)

        score = len(set(words)) / len(words)

        return score



