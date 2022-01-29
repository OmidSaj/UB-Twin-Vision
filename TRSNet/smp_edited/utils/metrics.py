from typing import Optional, List
import torch
# import torch.nn.functional as F
from . import base
from . import functional as F
from ..base.modules import Activation
from ._functional import soft_jaccard_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(
        self, 
        mode: str,
        average:bool = True,
        eps=1e-7, 
        smooth: float = 0.,
        threshold=0.5, 
        classes: Optional[List[int]] = None, 
        from_logits: bool = True,
        log_loss: bool = False,
        activation=None, 
        ignore_channels=None, 
        **kwargs):

        """Implementation of Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        
        super().__init__(**kwargs)
        
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)
            
        self.eps = eps
        self.average = average
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = torch.nn.functional.logsigmoid(y_pred).exp()
                
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = torch.nn.functional.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            
        scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims, threshold=self.threshold)

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        scores *= mask.float()


        if self.classes is not None:
            scores = scores[self.classes]

        if self.average:
            scores = scores.mean()
       
        return scores
        



class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pred, y_true):
        y_pred = self.activation(y_pred)
        return F.recall(
            y_pred, y_true,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


