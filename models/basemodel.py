import openai
import weave
from abc import ABC, abstractmethod
import traceback
import httpcore
import httpx
import time
from nltk.tokenize import sent_tokenize
import re

class ResponseChunkingError(Exception):
    def __init__(self, message, model_out, sentences=None, similarities=None, chunk_boarders=None):
        super().__init__(message)
        self.message = message
        self.model_out = model_out
        self.sentences = sentences
        self.similarities = similarities
        self.chunk_boarders = chunk_boarders


class BaseModel(ABC, weave.Model):
    model_name:str
    max_retries:int
    retry_wait_time:int
    split_method:str = "cosine" 
    last_line_min_len:int = 2

    @weave.op
    async def predict(self, system_prompt:str, user_prompt:str) -> dict:
        retries = 0
        errors = []
        model_output = ""
        paragraphs = {}
        while retries <= self.max_retries:
            try:
                model_output = await self.call_llm(system_prompt, user_prompt)
                if self.split_method == "cosine":
                    paragraphs = await self.extract_paragraphs(model_output, user_prompt)
                else:
                    paragraphs = await self.split_by_first_occurence(model_output, user_prompt)
                break
            except ValueError as e:
                errors.append({
                    "type":"ValueError",
                    "stacktrace":traceback.format_exc()
                })
                retries+=1
                paragraphs = {}
            except ResponseChunkingError as e:
                errors.append({
                    "type":"ResponseChunkingError",
                    "message":e.message,
                    "stacktrace":traceback.format_exc(),
                    "model_output":e.model_out,
                    "sentences":e.sentences,
                    "chunk_boarders":e.chunk_boarders,
                    "similarities":e.similarities
                })
                retries+=1
                paragraphs = {}
            except (httpcore.ReadTimeout, httpx.ReadTimeout, openai.APITimeoutError) as e:
                errors.append({
                    "type":"Readtimeout/APITimeout",
                    "stacktrace":traceback.format_exc(),
                })
                retries+=1
                paragraphs = {}
                time.sleep(self.retry_wait_time) # wait for a minute so there are less requests processed at a time by ollama
            except Exception as e:
                errors.append({
                    "type":e.__class__.__name__,
                    "stacktrace":traceback.format_exc()
                })
                retries+=1
                paragraphs = {}
        return {"split": paragraphs, "raw":model_output, "num_retries": retries, "errors": errors}

    @abstractmethod
    async def call_llm(self, system_prompt:str, user_prompt:str) -> str:
        pass

    @abstractmethod
    async def extract_paragraphs(self, model_output:str, user_prompt:str) -> dict:
        pass

    async def split_by_first_occurence(self, model_output:str, user_prompt:str):
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

                    if curr_topic is not None and curr_topic not in topic_split.keys():
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