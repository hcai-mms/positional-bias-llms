from models.basemodel import BaseModel, ResponseChunkingError
import weave
from abc import ABC
import traceback
import openai
import httpcore
import httpx
import time
import json
import logging

class BaseReevaluateModel(BaseModel, weave.Model):
    previous_responses_path:str
    _previous_responses:dict = {}
    
    async def call_llm(self, system_prompt:str, user_prompt:str) -> str:

        if len(self._previous_responses) == 0:
            print("Loading json file")
            start = time.time()
            with open(self.previous_responses_path, "r") as f:
                self._previous_responses = json.load(f)

            print(f"Json file loaded. Took: {time.time()-start:.2f}s")

        return self._previous_responses[user_prompt].pop(0)
