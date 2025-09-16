import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag

@scorer_library.register(name="named_entity_density")
class NamedEntityDensity(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.named_entity_density(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def named_entity_density(self, input: str) -> float:
        words = word_tokenize(input)

        tags = pos_tag(words)

        # NNP(S) = Proper Noun Singular (Plural)
        named_entities = [word for word, tag in tags if tag in ('NNP', 'NNPS')]

        score = len(named_entities) / len(words)
        return score