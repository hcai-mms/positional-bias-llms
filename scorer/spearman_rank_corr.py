import weave
from scorer.library import scorer_library
import numpy as np
from scipy import stats

@scorer_library.register(name="spearman")
class TopicSpearmanRankCorrlation(weave.Scorer):

    @weave.op
    async def score(self, output:dict, user_prompt:str) -> dict:
        topics = [up.strip() for up in user_prompt.split(",")]

        positions = []
        true_positions = []
        for i, topic in enumerate(topics):
            pos = output["split"][topic]["position"]
            if pos is not None:
                positions.append(pos)
                true_positions.append(i)

        return stats.spearmanr(true_positions, positions).statistic

