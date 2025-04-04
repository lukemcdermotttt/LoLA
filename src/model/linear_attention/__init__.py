"""
Linear and linear attention + sliding window classes
"""
from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState
)
from .linear_window_attention_tk import (
    LolcatsTKWindowAttention, LinearAttentionTKWindowCache
)
from .linear_window_attention_sw import (
    LolcatsSlidingWindowAttention, LinearAttentionSlidingWindowCache
)
# Experimental chunk linear attentions
from .linear_window_attention_tk_long import (
    LolcatsTKWindowLongAttention,
)
from .linear_window_attention_sw_long import (
    LolcatsSlidingWindowLongAttention,
)
from .linear_window_attention_tk_gen import (
    LolcatsWindowAttentionTKGen,
    LinearAttentionTKWindowGenerationCache
)

from .linear_window_attention_sw_sparse import (
    LolcatsSparseSlidingWindowAttention
)

from .linear_window_attention_sw_sparse_oracle import (
    LolcatsSparseSlidingWindowAttentionOracle
)

from .linear_window_attention_sw_h0 import (
    LolcatsSlidingWindowAttentionH0
)

from .linear_window_attention_sw_sparse_prefill import (
    LolcatsSparsePrefillSlidingWindowAttention, LinearAttentionSparseSlidingWindowCache
)