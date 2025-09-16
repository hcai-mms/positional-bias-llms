import weave
from scorer.library import scorer_library
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@scorer_library.register(name="topic_consistency")
class TopicConsistency(weave.Scorer):
    sementic_model_name:str = 'all-MiniLM-L6-v2'
    _semantic_model:SentenceTransformer = SentenceTransformer(sementic_model_name)

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.topic_consistency(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def topic_consistency(self, input: str) -> float:
        sentences = sent_tokenize(input)
        embeddings = self._semantic_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)

        score = np.mean(similarity_matrix[np.triu_indices(len(sentences), k=1)])
        return score.item()

