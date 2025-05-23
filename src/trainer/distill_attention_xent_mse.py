"""
Custom trainer class for distilling attentions ("attention transfer"). Can substitute for Hugging Face trainer.

In this implementation we support using either just the softmax attention outputs, or the softmax attention weights.
"""
import torch
import torch.nn as nn

from .default_lm import OurTrainer as DefaultTrainer


class OurTrainer(DefaultTrainer):
    """
    Custom trainer class for distilling attentions. 
    - We compute and store the attention outputs and/or weights for each head and layer,
      for both the "teacher" softmax attentions and "student" learnable subquadratic attentions
    - We then train the student layers to minimize either MSE(outputs) or CrossEntropy(weights)
    """
    def __init__(self,
                 model: nn.Module,
                 metric_for_best_model: str = 'distill/eval/loss',
                 mse_factor: float = 1e3,
                 xent_factor: float = 0,
                 **kwargs: any):
        super().__init__(model=model, 
                         metric_for_best_model=metric_for_best_model,
                         **kwargs)
        self.criterion_xent = nn.CrossEntropyLoss(reduction='mean')
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.mse_factor = mse_factor
        self.xent_factor = xent_factor
        self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss

    def compute_loss(self, model: nn.Module, data: dict[torch.Tensor],
                     sample_idx: int = None, **kwargs: any,) -> tuple[torch.Tensor, dict[any]]:
        """
        Attention distillation ("attention transfer")
        - For each layer and head, get attentions and train to 
          minimize some combo of MSE and cross-entropy loss
        """
        inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
        outputs = model(**inputs, output_attentions=True, use_cache=False)
        outputs = outputs.get('attentions')

        # Attentions are tuple[tuple[torch.Tensor, torch.Tensor]]
        # n_layers x (predicted_attns, true_attns)
        # predicted_attns and true_attns are shape (batch, n_heads, q_len, k_len)
        loss_mse = 0
        loss_xent = 0
        n_layers = 0  # Number of layers to distill
        softmax_layers = []
        for layer_idx, attns in enumerate(outputs):
            if attns is not None:
                print('attns', len(attns))
                if len(attns) != 2:
                    attns = attns.cpu()
                else:
                    if self.xent_factor > 0:
                        # Cross-entropy loss
                        a_pred, a_true = attns[0]
                        a_pred = a_pred.clamp(min=1e-12).log()  # nn.CrossEntropy assumes unnormalized logits
                        k_len = a_true.shape[-1]  # batch, n_heads, q_len, k_len
                        # Compute mean cross-entropy over all queries
                        a_pred = a_pred.contiguous().view(-1, k_len)
                        a_true = a_true.contiguous().view(-1, k_len)
                        loss_xent += self.criterion_xent(a_pred, a_true)
                    if self.mse_factor > 0:
                        print('mse',len(attns[1]), attns[1][0].size(), attns[1][1].size())
                        loss_mse += self.criterion_mse(*attns[1])
                    n_layers += 1
            else:
                softmax_layers.append(layer_idx)
        if n_layers > 0:
            loss_xent = loss_xent / n_layers * self.xent_factor
            loss_mse = loss_mse / n_layers * self.mse_factor
        loss = loss_xent + loss_mse
        if 'position_ids' in data:
            outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0,
                       'input_len': data['position_ids'].shape[1],
                       'position_ids': data['position_ids'][0].detach().cpu().numpy(),
                       'mse_factor': self.mse_factor,
                       'xent_factor': self.xent_factor,}
        else:
            outputs = {'loss_xent': loss_xent.item() if self.xent_factor > 0 else 0,
                       'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0, 
                       'mse_factor': self.mse_factor, 
                       'xent_factor': self.xent_factor}
        return loss, outputs
