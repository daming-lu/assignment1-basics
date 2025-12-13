import math

def cosine_lr_scheduler(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters
):
    """
    Cosine learning rate scheduler with warmup.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    cos_param = math.pi * ((it-warmup_iters) / (cosine_cycle_iters-warmup_iters))
    return min_learning_rate + 0.5 * (1+math.cos(cos_param)) * (max_learning_rate - min_learning_rate)