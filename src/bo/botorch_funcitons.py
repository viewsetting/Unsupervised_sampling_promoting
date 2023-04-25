from modulefinder import Module
import torch
from torch import Tensor
import torch.nn as nn
import math

from src.bo.botorch_fit import fit_gpytorch_torch
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.generation import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
import os
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
from botorch.test_functions.base import BaseTestProblem
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from torch import Tensor



def clamp_col(t:torch.tensor,bounds:torch.tensor):
    if len(bounds.shape) == 2:
        dim = bounds.shape[-1]
        for i in range(dim):
            t[:,i] = torch.clamp(t[:,i],min=bounds[0][i], max = bounds[1][i] )
    else:
        assert bounds.shape[0] == t.shape[0] #batch_size bounds
        dim = bounds.shape[-1]
        for i in range(dim):
            for j in range(bounds.shape[0]):
                t[j,i] = torch.clamp(t[j,i],min=bounds[j][0][i], max = bounds[j][1][i] )
        
    return t

def generate_batch_initial(bounds:torch.tensor, batch_size:int):
    """_summary_

    Args:
        bounds (tensor): _description_
        batch_size (int): _description_
    
    Returns:
        torch.tensor with shape (batch_size, dim)
    """
    
    dim = bounds.shape[-1]
    _initial = torch.zeros((batch_size,dim)).cuda()
    
    if len(bounds.shape) == 2:
        for idx in range(dim):
            _initial[:,idx].uniform_(bounds[0][idx],bounds[1][idx])
    else:
        bsz = bounds.shape[0]
        for idx in range(dim):
            for batch in range(bsz):
                _initial[batch,idx].uniform_(bounds[batch][0][idx],bounds[batch][1][idx])
    
    return _initial

def generate_batch_cadidates(initial_conditions:torch.tensor,
                             acquisition_function,
                             bounds:torch.tensor,
                             optimizer:torch.optim.Optimizer,
                             options: Optional[Dict[str, Union[float, str]]] = None,
):
    options = options or {}
    candidates = initial_conditions.requires_grad_(True)
    _optimizer = optimizer(params=[candidates],lr=options.get("lr",0.025))
    
    max_iter = int(options.get("max_iter",1))
    
    for idx in range(max_iter):
        with torch.no_grad():
            X = clamp_col(candidates,bounds)
        
        loss = -acquisition_function(X).sum()
        #print('[BO] [step] {} [loss] {}'.format(idx,loss))
        loss.requires_grad_(True)
        #print('loss shape',loss.shape)
        
        grad = torch.autograd.grad(loss, X,allow_unused=True)[0]
        
        def assign_grad():
            _optimizer.zero_grad()
            candidates.grad = grad
            return loss
        
        _optimizer.step(assign_grad)
    
    
    candidates = candidates.requires_grad_(False)
    final_candidates = clamp_col(candidates,bounds)
    
    return final_candidates


def one_step_BO(train_x,train_obj,bounds,max_iter:int=10,acq_factor:float=1.0,lr:float=0.1, acq_type:str='EI'):

    assert train_x.shape[0] == train_obj.shape[0]

    batch_size = train_x.shape[0]
    
    best_value,_ = torch.max(train_obj,dim=1)
    #print(best_value.shape)
    # train_x.retain_grad()
    # train_obj.retain_grad()
    #print('[train_obj] [requires_grad]',train_x.requires_grad,train_obj.requires_grad,train_obj.shape,train_x.shape)
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj).cuda()
    mll = ExactMarginalLogLikelihood(model.likelihood, model).cuda()
    # fit_gpytorch_torch(mll)
    mll.train()
    fit_gpytorch_torch(mll,options={'lr':lr,'maxiter':max_iter, 'disp':False})
    mll.eval()
    
    resampler = SobolQMCNormalSampler(num_samples=batch_size, seed=0, resample=True)        
#     MC_EI_resample = qExpectedImprovement(
#     model, best_f=best_value, sampler=resampler
# )
    if acq_type == 'EI':
        MC_EI_resample = XiqExpectedImprovement(
    model, best_f=best_value,xi=acq_factor, sampler=resampler
)
    elif acq_type == 'UCB':
        MC_EI_resample = qUpperConfidenceBound(model,beta=acq_factor,sampler=resampler)
    else:
        raise Exception
    #MC_EI_resample= qUpperConfidenceBound(model,beta=acq_factor,sampler=resampler)

    batch_initial_conditions = generate_batch_initial(bounds=bounds,
                       batch_size=batch_size
)


    batch_candidates = generate_batch_cadidates(
        initial_conditions=batch_initial_conditions,
        acquisition_function=MC_EI_resample,
        bounds=bounds,
        optimizer = torch.optim.Adam,
        options={'max_iter':max_iter, 'disp':False, 'lr':lr}
    )

    return batch_candidates


class XiqExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        xi: float=1.0,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.xi = xi

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)-self.xi).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei

class qUpperConfidenceBound(MCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.
    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)
    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.
        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = math.sqrt(beta * math.pi / 2)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.
        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)