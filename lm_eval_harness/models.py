"""
Inherit from lm-evaluation-harness/lm_eval/models/huggingface.py to load linearized models
"""
from typing import Tuple, Union, List, Any
import torch
import torch.nn.functional as F
from .models_huggingface import AutoCausalLM, AutoSeq2SeqLM
from src.model.modeling_llama import LolcatsLlamaForCausalLM as LOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LolcatsMistralForCausalLM as LOLCATS_MISTRAL_MODEL_CLASS

from src.model.modeling_llama import LooooolcatsLlamaForCausalLM as LOOOOOLCATS_LLAMA_MODEL_CLASS
from src.model.modeling_mistral import LooooolcatsMistralForCausalLM as LOOOOOLCATS_MISTRAL_MODEL_CLASS
from src.model.modeling_llama_sharded import ShardedLolcatsLlamaForCausalLM as SHARDED_LOLCATS_LLAMA_MODEL_CLASS



class LolcatsLlamaForCausalLM(AutoCausalLM):
    """
    Wrapper for Llama-like autoregressive language model
    """
    AUTO_MODEL_CLASS = LOLCATS_LLAMA_MODEL_CLASS
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

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], Any, Any]],
        disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """
        Compute the log likelihood for each (context, tokens) pair provided in requests.
        Each element of `requests` is a tuple:
            ((context: str, tokens: str), context_enc, tokens_enc)
        Returns a list of tuples (total_log_likelihood, is_greedy) for each request.
        """
        results = []
        for (context, tokens), _, _ in requests:
            # Concatenate context and tokens
            full_text = context + tokens
            input_ids = self.tok_encode(full_text)
            input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)  # [1, seq_len]

            with torch.no_grad():
                logits = self._model_call(input_tensor)  # [1, seq_len, vocab_size]

            # Shift logits for causal language modeling:
            # logits[:, t, :] predicts input_ids[t+1]
            logits = logits[:, :-1, :]
            target_ids = input_tensor[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Sum the log probabilities over all tokens
            total_log_likelihood = token_log_probs.sum().item()

            # Here we assume the model is greedy
            results.append((total_log_likelihood, True))
        return results

    def generate_until(self, context: str, until: Union[str, List[str]] = None, max_new_tokens: int = 100) -> str:
        # Ensure context is a string
        if not isinstance(context, str):
            context = str(context)

        # If no stop sequence is provided, use the tokenizer's eos token if available.
        if until is None:
            if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                until = self.tokenizer.eos_token
            else:
                raise ValueError("No stop sequence provided and tokenizer has no eos_token.")

        # Ensure until is a list of stop sequences
        if isinstance(until, str):
            stop_sequences = [until]
        else:
            stop_sequences = until

        # Tokenize the context
        token_context = self.tok_encode_batch([context])
        # Generate tokens using your model's generation method.
        generated_tokens = self._model_generate(
            inputs=token_context,
            max_tokens=max_new_tokens,
            stop=stop_sequences
        )
        # Decode tokens to text
        generated_text = self.tok_decode(generated_tokens.tolist())[0]
        # Truncate at the first occurrence of any stop sequence.
        for stop_seq in stop_sequences:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0]
                break
        return generated_text

    def loglikelihood_rolling(self, text: str, **kwargs) -> float:
        """
        Compute the rolling log likelihood for `text`. If the text length exceeds
        the model's max_length, process it in overlapping windows.
        """
        text = str(text)
        tokens = self.tok_encode(text)
        total_log_likelihood = 0.0

        # If the tokenized text fits within the max_length, process in one pass.
        if len(tokens) <= self.max_length:
            input_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self._model_call(input_tensor)
            logits = logits[:, :-1, :]
            target_ids = input_tensor[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.sum().item()

        # Otherwise, break text into overlapping windows.
        # Here, we use an overlap of 1 token to maintain continuity.
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
