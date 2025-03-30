"""
Inherit from lm-evaluation-harness/lm_eval/models/huggingface.py to load linearized models.
Using the new template LM interface (since the old “base LM” is gone).
"""
from typing import Tuple, Union, List, Any, Optional
import torch
import torch.nn.functional as F
from .models_huggingface import AutoCausalLM, AutoSeq2SeqLM
from src.model.modeling_llama import LolcatsLlamaForCausalLM as LOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LolcatsMistralForCausalLM as LOLCATS_MISTRAL_MODEL_CLASS

from src.model.modeling_llama import LooooolcatsLlamaForCausalLM as LOOOOOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LooooolcatsMistralForCausalLM as LOOOOOLCATS_MISTRAL_MODEL_CLASS
from src.model.modeling_llama_sharded import ShardedLolcatsLlamaForCausalLM as SHARDED_LOLCATS_LLAMA_MODEL_CLASS

from tqdm import tqdm  # ensure tqdm is installed
from .models_huggingface import stop_sequences_criteria
from lm_eval import utils

class LolcatsLlamaForCausalLM(AutoCausalLM):
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

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], Any, Any]],
        disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """
        Compute the log likelihood for each (context, tokens) pair.
        Each element of requests is a tuple:
            ((context: str, tokens: str), context_enc, tokens_enc)
        Returns a list of tuples (total_log_likelihood, is_greedy).

        This version mimics the original base LM behavior by:
          - Tokenizing context and answer separately.
          - Concatenating them, but then only summing the log probabilities for the answer tokens.
        """
        results = []
        for (context, tokens), _, _ in tqdm(requests, disable=disable_tqdm, desc="Evaluating loglikelihood"):
            # Tokenize context and answer separately.
            context_ids = self.tok_encode(context)
            answer_ids = self.tok_encode(tokens)
            full_ids = context_ids + answer_ids
            input_tensor = torch.tensor(full_ids, device=self.device).unsqueeze(0)  # [1, seq_len]

            with torch.no_grad():
                logits = self._model_call(input_tensor)  # [1, seq_len, vocab_size]

            # Compute log probs only for the answer tokens.
            # Note: logits[t] predicts token at position t+1.
            # The first answer token is at index len(context_ids),
            # and its prediction comes from logits at index len(context_ids)-1.
            answer_start = len(context_ids) - 1
            # We want predictions for answer tokens: logits from answer_start to second-last token.
            answer_logits = logits[:, answer_start:-1, :]
            # Corresponding target tokens (the answer tokens) start at index answer_start+1.
            target_ids = input_tensor[:, answer_start+1:]
            log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            total_log_likelihood = token_log_probs.sum().item()

            results.append((total_log_likelihood, True))
        return results

    def generate_until(self, context: str, until: Union[str, List[str]] = None, max_new_tokens: int = 100) -> str:
        # This method remains unchanged.
        if not isinstance(context, str):
            context = str(context)
        if until is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                until = self.tokenizer.eos_token
            else:
                raise ValueError("No stop sequence provided and tokenizer has no eos_token.")
        stop_sequences = [until] if isinstance(until, str) else until

        token_context = self.tok_encode_batch([context])
        generated_tokens = self._model_generate(
            inputs=token_context,
            max_tokens=max_new_tokens,
            stop=stop_sequences,
            use_cache=False,
            do_sample=False
        )
        generated_text = self.tok_decode(generated_tokens.tolist())[0]
        for stop_seq in stop_sequences:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0]
                break
        return generated_text

    def _model_generate(self, inputs, max_tokens: int, stop: Optional[List[str]] = None, **kwargs):
        # Ensure that the context does not encroach into the `space` for the generation.
        input_ids = inputs["input_ids"][:, -self.max_length:]
        attention_mask = inputs["attention_mask"][:, -self.max_length:]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        kwargs.pop("use_cache", None)
        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            use_cache=False,
            **kwargs
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )

    def loglikelihood_rolling(self, text: str, **kwargs) -> float:
        """
        Compute the rolling log likelihood for text.
        If the tokenized text fits within max_length, process it in one pass;
        otherwise, use overlapping windows.
        This method remains unchanged since it is typically used for full-text evaluation.
        """
        text = str(text)
        tokens = self.tok_encode(text)
        total_log_likelihood = 0.0

        if len(tokens) <= self.max_length:
            input_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self._model_call(input_tensor)
            logits = logits[:, :-1, :]
            target_ids = input_tensor[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum().item()

        step = self.max_length - 1
        for i in range(0, len(tokens), step):
            window_tokens = tokens[i : i + self.max_length]
            if len(window_tokens) < 2:
                break
            input_tensor = torch.tensor(window_tokens, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self._model_call(input_tensor)
            logits = logits[:, :-1, :]
            target_ids = input_tensor[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            total_log_likelihood += token_log_probs.sum().item()
        return total_log_likelihood

class LolcatsMistralForCausalLM(AutoCausalLM):
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


class ShardedLolcatsLlamaForCausalLM(AutoCausalLM):
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
        

class LooooolcatsLlamaForCausalLM(AutoCausalLM):
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


class LooooolcatsMistralForCausalLM(AutoCausalLM):
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

