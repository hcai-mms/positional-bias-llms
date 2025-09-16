import weave
from scorer.library import scorer_library
import zlib

@scorer_library.register(name="kolmogorov_complexity")
class KolmogorovComplexity(weave.Scorer):

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
        
        complexity = len(compressed_data)
        
        #? Normalize
        return complexity

