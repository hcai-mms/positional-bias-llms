import weave
from scorer.library import scorer_library
from transformers import pipeline, Pipeline


@scorer_library.register(name="emotion_analysis")
class EmotionAnalysis(weave.Scorer):
    emotion_classifier_model_name:str = "j-hartmann/emotion-english-distilroberta-base"
    _emotion_classifier:Pipeline = pipeline("text-classification", model=emotion_classifier_model_name, top_k=None)
    _max_model_length: int = _emotion_classifier.tokenizer.model_max_length

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.emotion_analysis(content["text"])
            for k, v in score.items():
                scores[f"{content["position"]}_{k}"] = v
                scores[f"{topic}_{k}"] = v
        return scores

    def emotion_analysis(self, input: str) -> float:
        model_inputs = self._emotion_classifier.preprocess(input)
        for k in model_inputs.keys():
            model_inputs[k] = model_inputs[k][:,:self._max_model_length]
        model_outputs = self._emotion_classifier.forward(model_inputs)
        out_size = model_outputs['logits'].size()[1]
        emotions = self._emotion_classifier.postprocess(model_outputs, top_k=out_size)
        score = {emotion['label']: emotion['score'] for emotion in emotions}
        return score