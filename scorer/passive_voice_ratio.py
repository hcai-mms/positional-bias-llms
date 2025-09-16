import weave
from scorer.library import scorer_library
from nltk import word_tokenize, sent_tokenize, pos_tag

@scorer_library.register(name="passive_voice_ratio")
class PassiveVoiceRatio(weave.Scorer):

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
        words = word_tokenize(input)
        tags = pos_tag(words)

        # VBN = Past Participle
        passive_count = sum(1 for i in range(len(tags) - 1) if tags[i][1] == 'VBN' and tags[i + 1][1] in ('IN', 'TO'))

        score = passive_count / len(words)
        return score