from models.reevaluate.library import model_reevaluate_library
from models.reevaluate.basemodel import BaseReevaluateModel
import weave
from models.basemodel import BaseModel
from models.gemini2flash_model import Gemini2Flash
from models.gemini15flash_model import Gemini15Flash
from models.gemini15flash8b_model import Gemini15Flash8B
from models.gemma22b_model import Gemma22B
from models.gemma227b_model import Gemma227B
from models.llama321B_model import Llama321BModel
from models.llama32_model import Llama32Model
from models.mistral_model import MistralModel
from models.phi3_mini import Phi3Mini
from models.phi3_medium import Phi3Medium
from models.qwen25_model import Qwen25Model
from models.qwen2532b_model import Qwen2532bModel
from models.solar_model import SolarModel
from models.wizardlm2_model import WizardLM2Model

@model_reevaluate_library.register(name="gemini2-flash")
class Gemini2FlashReevaluate(BaseReevaluateModel, Gemini2Flash, weave.Model):
    model_name:str = "gemini-2.0-flash-001"

@model_reevaluate_library.register(name="gemini1.5-flash")
class Gemini15FlashReevaluate(BaseReevaluateModel, Gemini15Flash, weave.Model):
    model_name:str = "gemini-1.5-flash-002"

@model_reevaluate_library.register(name="gemini1.5-flash-8b")
class Gemini15Flash8BReevaluate(BaseReevaluateModel, Gemini15Flash8B, weave.Model):
    model_name:str = "gemini-1.5-flash-8b-001"

@model_reevaluate_library.register(name="gemma2:2b")
class Gemma22BReevaluate(BaseReevaluateModel, Gemma22B, weave.Model):
    model_name:str = "gemma2:2b"

@model_reevaluate_library.register(name="gemma2:27b")
class Gemma227BReevaluate(BaseReevaluateModel, Gemma227B, weave.Model):
    model_name:str = "gemma2:27b"

@model_reevaluate_library.register(name="llama3.2")
class Llama32Reevaluate(BaseReevaluateModel, Llama32Model, weave.Model):
    model_name:str = "Llama3.2"

@model_reevaluate_library.register(name="llama3.2:1B")
class Llama321BReevaluate(BaseReevaluateModel, Llama321BModel, weave.Model):
    model_name:str = "Llama3.2:1B"

@model_reevaluate_library.register(name="mistral")
class MistralReevaluate(BaseReevaluateModel, MistralModel, weave.Model):
    model_name:str = "mistral"

@model_reevaluate_library.register(name="phi3-mini")
class Phi3MiniReevaluate(BaseReevaluateModel, Phi3Mini, weave.Model):
    model_name:str = "phi3:medium"

@model_reevaluate_library.register(name="phi3-medium")
class Phi3MediumReevaluate(BaseReevaluateModel, Phi3Medium, weave.Model):
    model_name:str = "phi3"

@model_reevaluate_library.register(name="qwen2.5")
class Qwen25Reevaluate(BaseReevaluateModel, Qwen25Model, weave.Model):
    model_name:str = "qwen2.5"

@model_reevaluate_library.register(name="qwen2.5:32b")
class Qwen2532bReevaluate(BaseReevaluateModel, Qwen2532bModel, weave.Model):
    model_name:str = "qwen2.5:32b"

@model_reevaluate_library.register(name="solar")
class SolarReevaluate(BaseReevaluateModel, SolarModel, weave.Model):
    model_name:str = "solar"

@model_reevaluate_library.register(name="wizardlm2")
class WizardLM2Reevaluate(BaseReevaluateModel, WizardLM2Model, weave.Model):
    model_name:str = "wizardlm2"

