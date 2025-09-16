import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag

@scorer_library.register(name="avg_word_len")
class AvgWordLen(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.average_word_length(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def average_word_length(self, input: str) -> float:
        words = [word for word in word_tokenize(input) if word.isalpha()]

        score = sum(len(word) for word in words) / len(words)
        return score