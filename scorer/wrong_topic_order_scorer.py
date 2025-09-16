import weave
from scorer.library import scorer_library
import numpy as np

@scorer_library.register(name="wrong_topic_order")
class WrongTopicOrderScorer(weave.Scorer):

    @weave.op
    async def score(self, output:dict, user_prompt:str) -> dict:
        topics = [up.strip() for up in user_prompt.split(",")]

        positions = []
        for i, topic in enumerate(topics):
            pos = output["split"][topic]["position"]
            positions.append(pos if pos else i)

        return (np.diff(positions) < 0).sum().item()

