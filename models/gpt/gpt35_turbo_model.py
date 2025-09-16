from models.basemodel import BaseModel, ResponseChunkingError
import re
import weave
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import convolve2d
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
from models.basemodel import BaseModel

import traceback
nltk.download('punkt_tab')

def read_gpt_resonse_json(path):
        gpt_out_df = pd.read_json(path, lines=True)
        gpt_out_df = pd.concat((gpt_out_df[["id","custom_id"]],
                            pd.json_normalize(gpt_out_df["response"])\
                                .add_prefix("response.")[["response.body.choices", 
                                                        "response.body.usage.prompt_tokens", 
                                                        "response.body.usage.completion_tokens",
                                                        "response.body.usage.total_tokens",
                                                        "response.body.model"
                                                        ]]
                        ), 
                        axis="columns")
        gpt_out_df["content"] = gpt_out_df["response.body.choices"].apply(lambda response: response[0]["message"]["content"])
        return gpt_out_df

class GPT35Turbo(BaseModel, weave.Model):
    model_name:str = "gpt-3.5-turbo-0125"
    sentence_transformer_model:str = 'all-MiniLM-L6-v2'
    _sentence_transformer:SentenceTransformer = SentenceTransformer(sentence_transformer_model)
    _header_examples:list[str] = [
        "Here are thorough descriptions for each item:",
        "Here are the descriptions of each item:",
        "Here are thedescriptions for each item:",
        "Here are detailed descriptions of each item on the list:"
    ]

    _footer_examples:list[str] = [
        "Please note that these descriptions are a simplified summary of each subject's significance and characteristics.",
        "These descriptions highlight not only the significance yet often overlooked nature of each topic but also the key players, initiatives, and consequences that have shaped their impact on human society.",
        "Please note that I provided a comprehensive overview of each item, highlighting their significance, background, and qualities, but this is just a small sample of the many facets and applications associated with these topics!",
        "These are just brief descriptions highlighting the importance, background, and qualities of each item."
    ]

    dataset_name:str = ""
    gpt_output_file:str = ""
    trials:int = 3
    split_method:str = "cosine"  
    last_line_min_len:int = 2
    _gpt_out_df:pd.DataFrame = pd.DataFrame()

    @weave.op()
    async def predict(self, id:int, system_prompt:str, user_prompt:str) -> dict:
        errors = []
        output = ""
        paragraphs = {}
        try:
            model_output = await self.call_llm(id, system_prompt, user_prompt)
            output = model_output["output"]
            if self.split_method == "cosine":
                paragraphs = self.extract_paragraphs(output,user_prompt)
            else:
                paragraphs = self.split_by_first_occurence(output, user_prompt)
        except Exception as e:
            errors.append({
                "type":e.__class__.__name__,
                "stacktrace":traceback.format_exc()
            })
            paragraphs = {}
        return {"split": paragraphs, "raw":output, "errors": errors}

    @weave.op()
    async def call_llm(self, id:int, system_prompt:str, user_prompt:str) -> str:
        if self._gpt_out_df.empty:
            print("loading json file")
            self._gpt_out_df = read_gpt_resonse_json(self.gpt_output_file)

        if self.trials > 1:
            for i in range(self.trials):
                custom_id = self.model_name + "-" + self.dataset_name + "-" + str(i) + "-" + str(id)
                row = self._gpt_out_df[self._gpt_out_df["custom_id"] == custom_id]
                if not row.empty:
                    self._gpt_out_df.drop(row.index, inplace=True)
                    break
        else:
            custom_id = self.model_name + "-" + self.dataset_name + "-" + id
            row = self._gpt_out_df[self._gpt_out_df["custom_id"] == custom_id]

        return {
            "usage": {
                "input_tokens": row["response.body.usage.prompt_tokens"].item(),
                "output_tokens": row["response.body.usage.completion_tokens"].item(),
                "total_tokens": row["response.body.usage.total_tokens"].item(),
            },
            "model": row["response.body.model"].item(),
            "output": row["content"].item(),
        }

    def extract_paragraphs(self, model_output:str, user_prompt:str) -> dict:
        paragraphs = {}
        topics = user_prompt.split(",")
        
        topics = [t.strip() for t in topics]
        lines = model_output.split("\n\n")
        sentences = []
        for line in lines:
            line = re.sub(r"[*#]", "", line).strip()
            sent_tokens = sent_tokenize(line)
            sentences.extend(sent_tokens)

        for s in self._header_examples:
            sentences.insert(0,s)
        for s in self._footer_examples:
            sentences.append(s)

        text_embedding = self._sentence_transformer.encode(sentences)
        
        topic_embedding = self._sentence_transformer.encode(topics)

        similarities = cosine_similarity(text_embedding[:10])
        diag_similarities = similarities.diagonal(1)

        topic_sent_sim = cosine_similarity(text_embedding, topic_embedding).T
        topic_sent_sim_ma = moving_average_for_topics(topic_sent_sim,3)
        missing_topic_mask = topic_sent_sim_ma.max(axis=1) < 0.5
        topic_sent_sim[missing_topic_mask] = 0
        
        topic_sent_max = topic_sent_sim_ma.argmax(axis=0)
        header_end = np.where(diag_similarities < 0.7)[0][0]
        topic_sent_max[:max(header_end,4)] = -1 # max to ensure our added headers are never included

        last_valid_idx = len(topic_sent_max)-2

        topic_locations = []
        for topic_id in range(len(topics)):
            location = topic_sent_max == topic_id
            if np.abs(np.diff(location)).sum() > 2:
                topic_matches_indices = np.nonzero(location)[0]
                start_idx = topic_matches_indices[0]
                last_idx = start_idx
                current_seq = [start_idx,]
                longest_seqence = list()

                for topic_match_idx in topic_matches_indices[1:]:
                    if topic_match_idx - last_idx < 4: # allow for 1-2 missed spot, to avoid cutting the topic sections short
                        current_seq.append(topic_match_idx)
                    else:
                        if len(longest_seqence) < len(current_seq):
                            longest_seqence = current_seq
                        current_seq = [topic_match_idx,]
                    last_idx = topic_match_idx

                if len(longest_seqence) < len(current_seq):
                    longest_seqence = current_seq
                # +1 and +2 to compensate for the offset
                # min so we never include the added footers
                topic_locations.append([int(longest_seqence[0])+1, min(int(longest_seqence[-1])+2,last_valid_idx)]) 
            else:
                location_list = (np.where(np.diff(location) != 0)[0]+2).tolist()
                if len(location_list) == 2:
                    location_list[1] = min(location_list[1],last_valid_idx) # min so we never include the added footers
                elif len(location_list) == 1:
                    location_list.append(last_valid_idx)
                topic_locations.append(location_list)

        paragraphs = dict()

        text_topic_ids = np.arange(len(topics))

        last_end = 0
        for i, s in enumerate(topic_locations):
            if len(s) == 2:
                text_topic_ids[i] = s[0]
                last_end = s[1]+1
            else:
                text_topic_ids[i] = last_end
                last_end += 1

        text_topic_ids = np.argsort(text_topic_ids)

        for i, topic in enumerate(topics):
            paragraph = ""
            topic_id = np.where(text_topic_ids == i)[0]

            if topic_id.size == 1:
                topic_id = topic_id.item()
                if len(topic_locations[i]) == 2:
                    for k in range(*topic_locations[i]):
                        sentence = sentences[k].replace("*","") # remove bold markdown 
                        sentence = sentence.strip()
                        if i == 0 and sentence.startswith("Here are"):
                            continue
                        if len(sentence) > 3: # filter out really short sections (e.g. section numbering like '1.')
                            paragraph += " " + sentence
                    paragraph = paragraph.strip()
                    paragraphs[topic] = {
                                "position": topic_id,
                                "text":paragraph
                            }
                    continue
            paragraphs[topic] = {
                        "position": None,
                        "text": None
                    }

        return paragraphs
    
    def split_by_first_occurence(self, model_output:str, user_prompt:str):
        topics = user_prompt.split(", ")

        lines = model_output.split("\n\n")
        num_lines = len(lines)
        sentences = []
        for i, line in enumerate(lines):
            line = re.sub(r"\*|#|(-\s)|(\d+\.\s)","", line).strip()
            sent_tokens = sent_tokenize(line)
            if i+1 == num_lines and len(sent_tokens) < self.last_line_min_len:
                continue
            # sent_tokens = [s for s in sent_tokens if len(re.sub(r"\d\.?", "", s))>3]
            sentences.extend(sent_tokens)

        curr_topic = None
        topic_start = -1

        topic_split = {}

        for i, sent in enumerate(sentences):
            for topic in topics:
                # all parts of the topic need to be in the sentence
                # this is so we find match e.g. topic = "James Webb Telescope"
                # while sent="James Webb Space Telescope"
                if all([t_part.lower() in sent.lower() for t_part in topic.split(" ")]) \
                    and topic != curr_topic:
                    
                    if curr_topic and curr_topic not in topic_split.keys():
                        topic_split[curr_topic] = (topic_start, i, len(topic_split))

                    curr_topic = topic
                    topic_start = i
        topic_split[curr_topic] = (topic_start, len(sentences), len(topic_split))

        paragraphs = dict()

        for topic, (start, end, pos) in topic_split.items():
            paragraphs[topic] = {
                            "position": pos,
                            "text": ' '.join(sentences[start:end])
                        }
        
        if len(topics) != len(paragraphs):
            for topic in topics:
                if topic not in paragraphs.keys():
                    paragraphs[topic] = {
                            "position": None,
                            "text": None
                        }
        return paragraphs

    
def moving_average_for_topics(x, w):
    return convolve2d(x, np.pad(np.ones(3).reshape(1,-1), pad_width=((1,1),(0,0))), 'same', boundary='fill')[:,1:-1] / w