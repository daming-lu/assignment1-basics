import math
def gradient_clipping(parameters, max_l2_norm):
    """
    Clip the gradients of the model parameters to the specified maximum norm.
    """
    total_norm = 0
    
    for p in parameters:
        if p.grad is None:
            continue
        cur_norm = p.grad.data.norm(2)
        total_norm += cur_norm ** 2
    
    total_norm = total_norm ** 0.5
    if total_norm >= max_l2_norm:
        coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= coef
    