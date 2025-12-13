# import torch.optim.Optimizer
import torch
import torch.optim

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)
                    
        # super().__init__(params, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        # import ipdb;ipdb.set_trace()
        for group in self.param_groups:
            # print('group:', group)
            """
            {
            "params":[
                "Parameter containing":"tensor("[
                    [0.4414,0.4792,-0.1353],
                    [0.5304,-0.1265,0.1165]
                ], "requires_grad=True)"
            ],
            "lr":0.001,
            "betas":(0.9,0.999),
            "eps":1e-08,
            "weight_decay":0.01
            }
            """
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]   
            
            for p in group["params"]:
                # print('p:', p)
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)  # m
                    state['exp_avg_sq'] = torch.zeros_like(p)  # v
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update m
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update v
                # beta2_numerator = (beta2 ** state['step'])**0.5
                # beta1_denom = 1 - beta1 ** state['step']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # Compute adjusted learning rate: αt = α * sqrt(1 - β2^t) / (1 - β1^t)
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                # lr.mul_(step_size)
                denom = exp_avg_sq.sqrt().add_(eps)
                # grad.add_(step_size, value=-exp_avg.div_(denom))
                p.addcdiv_(exp_avg, denom, value=-step_size)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss
        