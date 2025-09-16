import weave
from scorer.library import scorer_library

@scorer_library.register(name="missing_topic")
class MissingTopicScorer(weave.Scorer):

    @weave.op
    async def score(self, output:dict, user_prompt:str) -> dict:
        scores = {}
        out_topics = [topic for topic, content in output["split"].items() if content["text"] is not None]
        for pos, topic in enumerate(user_prompt.split(",")):
            topic = topic.strip()
            if topic in out_topics:
                scores[f"{pos}"] = 0
                scores[f"{topic}"] = 0
            else:
                scores[f"{pos}"] = 1
                scores[f"{topic}"] = 1

        return scores

