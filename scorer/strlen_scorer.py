import weave
import weave.trace
import weave.trace.vals
from scorer.library import scorer_library

@scorer_library.register(name="strlen")
class StrLenScorer(weave.Scorer, weave.Object):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = len(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores