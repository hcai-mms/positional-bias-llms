import weave
from scorer.library import scorer_library
from nltk import word_tokenize

@scorer_library.register(name="cohesive_device_density")
class CohesiveDeviceDensity(weave.Scorer):
    # All connective words according to https://www.grammarbank.com/connectives-list.html
    cohesive_words: list[str] = {
        "and", "also", "besides", "further", "furthermore", "too", "moreover", "in addition", "then", "of equal importance", 
        "equally important", "another", "next", "afterward", "finally", "later", "last", "lastly", "at last", "now", 
        "subsequently", "when", "soon", "thereafter", "after a short time", "the next week", "the next month", "the next day", 
        "a minute later", "in the meantime", "meanwhile", "on the following day", "at length", "ultimately", "presently", 
        "first", "second", "finally", "hence", "from here on", "to begin with", "last of all", "after", "before", "as soon as", 
        "in the end", "gradually", "above", "behind", "below", "beyond", "here", "there", "to the right", "to the left", 
        "nearby", "opposite", "on the other side", "in the background", "directly ahead", "along the wall", "as you turn right", 
        "at the top", "across the hall", "at this point", "adjacent to", "for example", "to illustrate", "for instance", 
        "to be specific", "such as", "just as important", "similarly", "in the same way", "as a result", "hence", "so", "accordingly", 
        "as a consequence", "consequently", "thus", "since", "therefore", "for this reason", "because of this", "to this end", 
        "for this purpose", "with this in mind", "for this reason", "like", "in the same manner", "as so", "similarly", "but", 
        "in contrast", "conversely", "however", "still", "nevertheless", "nonetheless", "yet", "and yet", "on the other hand", 
        "on the contrary", "or", "in spite of this", "actually", "in fact", "in summary", "to sum up", "to repeat", "briefly", 
        "in short", "finally", "on the whole", "therefore", "as I have said", "in conclusion", "as you can see"
    }

    @weave.op
    async def score(self, output:dict) -> dict:
        scores = {}
        for topic, content  in output["split"].items():
            if content["text"] is None or len(content["text"]) == 0:
                continue
            score = self.cohesive_device_density(content["text"])
            scores[f"{content["position"]}"] = score
            scores[f"{topic}"] = score
        return scores

    def cohesive_device_density(self, input: str) -> float:
        words = [word.lower() for word in word_tokenize(input) if word.isalpha()]

        cohesive_count = sum(1 for word in words if word in self.cohesive_words)

        score = cohesive_count / len(words)
        return score