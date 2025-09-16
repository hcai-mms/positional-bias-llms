import weave
from scorer.library import scorer_library
from nltk import sent_tokenize

@scorer_library.register(name="question_density")
class QuestionDensity(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.question_density(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def question_density(self, input: str) -> int:
        sentences = sent_tokenize(input)

        question_count = sum(1 for sentence in sentences if sentence.strip().endswith('?'))

        score = question_count / len(sentences)
        return score

