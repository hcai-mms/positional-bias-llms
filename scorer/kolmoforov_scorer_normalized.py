import weave
from scorer.library import scorer_library
import zlib

@scorer_library.register(name="kolmogorov_complexity_normalized")
class KolmogorovComplexityNormalized(weave.Scorer):

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.kolmogorov_complexity(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def kolmogorov_complexity(self, input_string: str) -> int:
        input_bytes = input_string.encode('utf-8')
    
        compressed_data = zlib.compress(input_bytes)
        
        normalized_complexity = len(compressed_data) / len(input_string)
        
        return normalized_complexity

