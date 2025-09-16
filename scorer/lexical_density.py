import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag

@scorer_library.register(name="lexical_density")
class LexicalDensity(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.lexical_density(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def lexical_density(self, input: str) -> float:
        words = [word for word in word_tokenize(input) if word.isalpha()]

        # Nouns, Verbs, Adjectives, Adverbs
        content_words = [word for word, pos in pos_tag(words) if pos.startswith(('N', 'V', 'J', 'R'))]

        score = len(content_words) / len(words)
        return score