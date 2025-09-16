import weave
from scorer.library import scorer_library
import math
from collections import Counter

@scorer_library.register(name="shannon_entropy_normalized")
class ShanonEntropyNormalized(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.shannon_entropy(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def shannon_entropy(self, input: str) -> float:
        char_counts = Counter(input)
    
        total_chars = len(input)
        
        entropy = 0.0
        for char, count in char_counts.items():
            p_i = count / total_chars
            entropy -= p_i * math.log2(p_i)
        
        max_entropy = math.log2(len(char_counts)) if len(char_counts) > 1 else 1

        normalized_entropy = entropy / max_entropy
        return normalized_entropy

