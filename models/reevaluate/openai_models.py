from models.gpt.gpt4o_mini_model import GPT4oMini
from models.gpt.gpt4o_model import GPT4o
from models.gpt.gpt35_turbo_model import GPT35Turbo
from models.reevaluate.basemodel import BaseReevaluateModel
from models.reevaluate.library import model_reevaluate_library
import weave

@model_reevaluate_library.register(name="gpt-4o-mini-2024-07-18")
class GPT4oMiniReevaluate(BaseReevaluateModel, GPT4oMini, weave.Model):
    model_name:str = "gpt-4o-mini-2024-07-18"

    async def call_llm(self, id:int, system_prompt:str, user_prompt:str) -> str:
        output = await BaseReevaluateModel.call_llm(self, system_prompt, user_prompt)
        return {"output":output}

@model_reevaluate_library.register(name="gpt-4o-2024-11-20")
class GPT4oReevaluate(BaseReevaluateModel, GPT4o, weave.Model):
    model_name:str = "gpt-4o-2024-11-20"
    
    async def call_llm(self, id:int, system_prompt:str, user_prompt:str) -> str:
        output = await BaseReevaluateModel.call_llm(self, system_prompt, user_prompt)
        return {"output":output}

@model_reevaluate_library.register(name="gpt-3.5-turbo-0125")
class GPT35TurboReevaluate(BaseReevaluateModel, GPT35Turbo, weave.Model):
    model_name:str = "gpt-3.5-turbo-0125"
    
    async def call_llm(self, id:int, system_prompt:str, user_prompt:str) -> str:
        output = await BaseReevaluateModel.call_llm(self, system_prompt, user_prompt)
        return {"output":output}