import weave
from typing import Any
from collections import defaultdict
from scorer.library import scorer_library
from models.library import model_library
import evaluate
import numpy as np
import time
import json

@scorer_library.register(name="golden_description_rouge")
class GoldenDescriptionROUGEScorer(weave.Scorer):
    golden_descritions:str
    _golden_descritions:dict = {}

    @weave.op
    async def score(self, output:dict, user_prompt:str) -> dict:

        if len(self._golden_descritions) == 0:
            print("Loading ROUGE json file")
            start = time.time()
            with open(self.golden_descritions, "r") as f:
                self._golden_descritions = json.load(f)

            print(f"ROUGE Json file loaded. Took: {time.time()-start:.2f}s")

        scores = {}
        for topic, content  in self._golden_descritions[user_prompt].pop(0).items():
            if content["position"] is None:
                continue
            score = content["score"]
            # score = self.golden_rouge(content["text"], topic)
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def golden_rouge(self, input: str, topic) -> float:
        results_individual = [self._rouge.compute(predictions=[input], references=[ref]) for ref in self.golden_descritions[topic]]
        
        results = {'rouge1':[], 'rouge2':[], 'rougeL':[], 'rougeLsum':[]}
        for res in results_individual:
            for k, v in res.items():
                results[k].append(v)
        
        return {k:np.mean(v) for k,v in results.items()}
