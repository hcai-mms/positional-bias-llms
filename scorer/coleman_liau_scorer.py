import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize

@scorer_library.register(name="coleman_liau_index")
class ColemanLiauIndex(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.coleman_liau_index(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def coleman_liau_index(self, input: str) -> float:
        sentences = sent_tokenize(input)
        words = word_tokenize(input)

        num_characters = sum(len(word) for word in words if word.isalpha())
        num_words = len([word for word in words if word.isalpha()])

        L = (num_characters / num_words) * 100
        S = (len(sentences) / num_words) * 100

        score = 0.0588 * L - 0.296 * S - 15.8

        return score

