import torch
import numpy as np

from torch.optim.optimizer import Optimizer, required


class Acc_SNGD(Optimizer):
    r"""
    Accelerated and stochastic version for NGD
    """
    def __init__(self, params, lr=1e-4, eta0=0.5, eta1=0.45, alpha=4, gamma=0.9, nesterov=False, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid initial learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(gamma=gamma, lr=lr, k=0, eta0=eta0, eta1=eta1, alpha=alpha, nesterov=nesterov, weight_decay=weight_decay)
        super(Acc_SNGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Acc_SNGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-4)
            group.setdefault('k', 0)
            group.setdefault('eta0', 0.5)
            group.setdefault('eta1', 0.45)
            group.setdefault('alpha', 4)
            group.setdefault('gamma', 0.9)

    def compute_dif_norms(self, prev_optimizer=required):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)

    def compute_lr(self, k, lr, grad_dif_norm, param_dif_norm, eta0, eta1, alpha):
        if k == 0:
            lr_new = lr
        else:
            if grad_dif_norm > (eta0 / lr) * param_dif_norm:
                lr_new = eta1 * param_dif_norm / grad_dif_norm
            else:
                eps = (1/k) ** alpha
                lr_new = (1 + eps) * lr

        return lr_new

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr_new = self.compute_lr(k=group['k'], lr=group['lr'], grad_dif_norm=group['grad_dif_norm'],
                                     param_dif_norm=group['param_dif_norm'], eta0=group['eta0'], eta1=group['eta1'], alpha=group['alpha'])
            group['lr'] = lr_new
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p = d_p.add(group['weight_decay'], p.data)
                if group['gamma'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['gamma']).add_(d_p, alpha=1)
                    if group['nesterov']:
                        d_p = d_p.add(buf, alpha=group['gamma'])
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha=-lr_new)

            group['k'] = group['k'] + 1
        return loss

class SNGD(Optimizer):
    r"""
    Stochastic version for NGD
    """
    def __init__(self, params, lr=1e-3, eta0=0.4, eta1=0.35, beta=4, alpha=3, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid initial learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, k=0, eta0=eta0, eta1=eta1, beta=beta, alpha=alpha, weight_decay=weight_decay)
        super(SNGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SNGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-3)
            group.setdefault('k', 0)
            group.setdefault('eta0', 0.4)
            group.setdefault('eta1', 0.35)
            group.setdefault('beta', 4)
            group.setdefault('alpha', 3)

    def compute_dif_norms(self, prev_optimizer=required):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            k = group['k']
            eta0 = group['eta0']
            eta1 = group['eta1']
            beta = group['beta']
            alpha = group['alpha']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            if k == 0:
                lr_new = lr
            else:
                if grad_dif_norm > (eta0 / lr) * param_dif_norm:
                    lr_new = eta1 * param_dif_norm / grad_dif_norm
                else:
                    eps = (np.log(k) ** beta) / (k ** alpha)
                    lr_new = (1 + eps) * lr
            group['lr'] = lr_new
            group['k'] = k + 1
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                p.data.add_(d_p, alpha=-lr_new)

        return loss

class Adsgd(Optimizer):
    r"""
    Adaptive SGD with estimation of the local smoothness (curvature).
    Based on https://arxiv.org/abs/1910.09529
    """
    def __init__(self, params, lr=0.2, amplifier=0.02, theta=1, damping=1, eps=1e-5, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid initial learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, amplifier=amplifier, theta=theta, damping=damping,
                        eps=eps, weight_decay=weight_decay)
        super(Adsgd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adsgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 0.2)
            group.setdefault('amplifier', 0.02)
            group.setdefault('damping', 1)
            group.setdefault('theta', 1)
                
    def compute_dif_norms(self, prev_optimizer=required):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        # TODO: use closure to compute gradient difference
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            damping = group['damping']
            amplifier = group['amplifier']
            theta = group['theta']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            if param_dif_norm > 0 and grad_dif_norm > 0:
                lr_new = min(lr * np.sqrt(1 + amplifier * theta), param_dif_norm / (damping * grad_dif_norm)) + eps
            else:
                lr_new = lr * np.sqrt(1 + amplifier * theta)
            theta = lr_new / lr
            group['theta'] = theta
            group['lr'] = lr_new
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                p.data.add_(d_p, alpha=-lr_new)
        return loss
