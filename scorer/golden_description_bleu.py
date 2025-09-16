import weave
import weave.trace
from scorer.library import scorer_library
import evaluate
import numpy as np

@scorer_library.register(name="golden_description_bleu")
class GoldenDescriptionBLEUScorer(weave.Scorer):
    golden_descritions:dict
    _bleu = evaluate.load("bleu")

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.golden_bleu(content["text"], topic)
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def golden_bleu(self, input: str, topic:str) -> float:
        results_individual = [self._bleu.compute(predictions=[input], references=[ref]) for ref in self.golden_descritions[topic]]
        results = {'bleu':[], 'brevity_penalty':[], 'length_ratio':[], 'translation_length':[], 
                   'reference_length':[], "precisions_1":[], "precisions_2":[], "precisions_3":[], "precisions_4":[],}
        for res in results_individual:
            for k, v in res.items():
                if k == "precisions":
                    for i, p_v in enumerate(v):
                        results[f"precisions_{i+1}"].append(p_v)
                    continue
                results[k].append(v)
        
        return {k:np.mean(v) for k,v in results.items()}
