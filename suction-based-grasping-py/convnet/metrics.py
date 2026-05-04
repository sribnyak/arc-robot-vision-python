import torch
import torch.nn.functional as F


def masked_bce_loss(logit_map: torch.Tensor, target: torch.Tensor, ignore_class: int = 2) -> torch.Tensor:
    """
    Calculate cross-entropy loss criterion, ignoring specified class.
    
    Args:
        logit_map: Predicted pixel-wise logit map, a tensor of shape (batch, 1, height, width)
        target: Target segmentation map, an integer-valued tensor of shape (batch, 3, height, width)
        ignore_class: Index of the class that will have weight 0.
        
    Returns:
        Tensor loss value
    """

    logit_map = logit_map.squeeze(1)  # (batch, h, w)
    valid_mask = (target != ignore_class)

    if valid_mask.any():
        valid_logits = logit_map[valid_mask]  # (batch,)
        valid_targets = target[valid_mask]    # (batch,)

        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets.float())
    else:  # no valid pixels in batch
        return torch.tensor(0., device=logit_map.device, requires_grad=True)


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, ignore_class: int = 2) -> torch.Tensor:
    """
    Calculate MSE loss criterion, ignoring specified class.

    Args:
        prediction: Predicted pixel-wise affordances, a tensor of shape (batch, 1, height, width)
        target: Target segmentation map, an integer-valued tensor of shape (batch, 3, height, width)
        ignore_class: Index of the class that will have weight 0.
        
    Returns:
        Tensor loss value
    """

    prediction = prediction.squeeze(1)  # (batch, h, w)
    valid_mask = (target != ignore_class)

    if valid_mask.any():
        valid_preds = prediction[valid_mask]  # (batch,)
        valid_targets = target[valid_mask]    # (batch,)

        return F.mse_loss(valid_preds, valid_targets.float(), reduction='mean')
    else:  # no valid pixels in batch
        return torch.tensor(0., device=prediction.device, requires_grad=True)
