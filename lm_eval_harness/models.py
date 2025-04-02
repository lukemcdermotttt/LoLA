"""
Inherit from lm-evaluation-harness/lm_eval/models/huggingface.py to load linearized models.
Using the new template LM interface (since the old “base LM” is gone).
"""
from typing import Tuple, Union, List, Any, Optional
import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
#from .models_huggingface import AutoCausalLM, AutoSeq2SeqLM
from src.model.modeling_llama import LolcatsLlamaForCausalLM as LOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LolcatsMistralForCausalLM as LOLCATS_MISTRAL_MODEL_CLASS

from src.model.modeling_llama import LooooolcatsLlamaForCausalLM as LOOOOOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LooooolcatsMistralForCausalLM as LOOOOOLCATS_MISTRAL_MODEL_CLASS
from src.model.modeling_llama_sharded import ShardedLolcatsLlamaForCausalLM as SHARDED_LOLCATS_LLAMA_MODEL_CLASS

from tqdm import tqdm  # ensure tqdm is installed
#from .models_huggingface import stop_sequences_criteria
from lm_eval import utils

class LolcatsLlamaForCausalLM(HFLM):
    """
    Wrapper for Llama-like autoregressive language model.
    This implementation mimics the old base LM behavior by computing loglikelihoods only on the answer (completion) tokens.
    """
    AUTO_MODEL_CLASS = LOLCATS_LLAMA_MODEL_CLASS

    @property
    def add_special_tokens(self) -> bool:
        """Determines whether special tokens are added."""
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class LolcatsMistralForCausalLM(HFLM):
    """
    Wrapper for Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOLCATS_MISTRAL_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class ShardedLolcatsLlamaForCausalLM(HFLM):
    """
    Wrapper for Llama or Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = SHARDED_LOLCATS_LLAMA_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


# class ShardedRollLolcatsLlamaForCausalLM(AutoCausalLM):
#     """
#     Wrapper for Llama or Mistral-like autoregressive language model
#     """
#     AUTO_MODEL_CLASS = SHARDED_ROLL_LOLCATS_LLAMA_MODEL_CLASS
#     @property
#     def add_special_tokens(self) -> bool:
#         """Whether to include special tokens in encoded text. This should be
#         determined by whether or not the model was trained with special tokens.
#         TODO: Remove these conditionals once HuggingFace supports a way to
#         check whether or not an arbitrary model was trained with special tokens.
#         """
#         if self._add_special_tokens is not None:
#             return self._add_special_tokens
#         else:
#             return False
        

class LooooolcatsLlamaForCausalLM(HFLM):
    """
    Wrapper for Llama-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOOOOOLCATS_LLAMA_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False


class LooooolcatsMistralForCausalLM(HFLM):
    """
    Wrapper for Mistral-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOOOOOLCATS_MISTRAL_MODEL_CLASS
    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        else:
            return False

