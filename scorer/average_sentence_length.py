import weave
from scorer.library import scorer_library
from nltk.tokenize import word_tokenize, sent_tokenize

@scorer_library.register(name="avg_sent_len")
class AverageSentenceLength(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.average_sentence_length(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def average_sentence_length(self, input: str) -> float:
        sentences = sent_tokenize(input)
        num_words = sum(len(word_tokenize(sentence)) for sentence in sentences)

        score = num_words / len(sentences)

        return score

