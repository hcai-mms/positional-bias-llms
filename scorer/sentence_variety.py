import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag

@scorer_library.register(name="sentence_variety")
class SentenceVariety(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.sentence_variety(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def sentence_variety(self, input: str) -> float:
        sentences = sent_tokenize(input)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

        score = max(sentence_lengths) - min(sentence_lengths)
        return score