
import math
import torch
from gmake_lpf import *

def _pprt(t, x_txt, x):
    '''Print signal, x'''
    if x > 0: print(t, x_txt, x)

# =========================================================
# Helper utilities
# =========================================================
class utl():
    @staticmethod
    def sq(x):
        """
        Element-wise square.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Element-wise squared values.
        """
        return x * x
    
    @staticmethod
    def frt(x):
        """
        Element-wise fourth power.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Element-wise values to power 4.
        """
        return x * x * x * x
    
    @staticmethod
    def ompow(rho, t):
        """
        Trsnsient bias-correction factor: 1 - rho^t, 0 <= x < 1.

        Used to remove transient bias.

        Args:
            rho (float or Tensor): filter coefficient.
            t (int or Tensor): iteration index.

        Returns:
            Tensor: Bias-correction denominator.
        """
        return 1 - torch.pow(rho, t)    

    @staticmethod
    def classdim(G, usemat=True, mn_dim=4, mx_dim=2048):
        """
        classify tensor G to use matrix 2D update or not.
        Args:
            G (Tensor)
            usemat (bool, optional): _flag_. Defaults to True.
            mn_dim (int, optional): _small_dim_. Defaults to 4.
            mx_dim (int, optional): _large_dim_. Defaults to 2048.
        Returns:
            tuple: _(classG, reshaped, true_shape)_. if _classG_ is True, use 2d update; _reshaped_ is True if G reshaped to 2d;
            _true_shape_ is the original shape of G.
        """

        true_shape = G.shape
        # print('in',G.shape)

        # vector-like cases
        casevec = (not usemat) or G.is_sparse or (G.ndim==1)
        if casevec: return False, False, true_shape

        # if usemat...
        # test-case [not used, see casevec]
        case1dim = (G.ndim == 1)
        if case1dim:
            noflag = G.shape[0] < mn_dim or G.shape[0] > mx_dim
            if noflag:  
                # print('nop', G.shape)
                return False, False, true_shape
            # print('yup', G.shape) 
            return True, True, true_shape

        # reshape to a matrix if ndim > 2
        reshaped = False
        if G.ndim > 2: 
            G = G.reshape(G.shape[0], -1)
            reshaped = True
            
        n, m = G.shape
        mn, mx = (n, m) if n < m else (m, n)
        # mn too small or mx too large
        noflag = mn < mn_dim  or mx > mx_dim
        if noflag: return False, False, true_shape
        
        # print('out', G.shape)
        return True, reshaped, true_shape

# =========================================================
# Window function
# =========================================================
class VTRWin():
    """
    Optimal First-order Variational Window Function.

    Defines a parametric family of window functions over normalized time,
    used to modulate update magnitudes multiplicatively.

    The construction supports:
        - Linear and cosine envelopes
        - Square-root deformation
        - Peak shifting and flat-top shaping

    Args:
        optn (int): Window type selector:
            0: inactive [unit constant]
            1: linear decay,
            2: raised cosine 
        p (int): p-moment power for (1/p), set p=2 for sqrt forms.
        m (float): Peak location in normalized time.
        e (float): Flat-top width parameter.
        l (float): Minimum output value.
        T (int): Window length or period.

    Notes:
        Normalized time is defined as:
            x = t' / (T - 1),  with t' ∈ [0, T-1]
    """

    def __init__(self, optn=0, p=1, m=0.0, e=0.0, l=0.0, T=1e9):
        self.optn = optn
        self.pinv = 1/math.fabs(p)
        self.m = m
        self.e = e
        self.low = l
        self.T = T if T >= 3 else 3

    def step(self, t):
        """
        Evaluate window at iteration t.

        Args:
            t (int or Tensor): Iteration index (1-based).

        Returns:
            Tensor: Window value in [low, 1].
        """
        if self.optn == 0: return 1

        x = torch.remainder(t - 1, self.T) / (self.T - 1)
        x = self.tfx(x, self.m, self.e)

        if self.optn == 1: y = self.flin(x)
        elif self.optn == 2: y = self.frcos(x)
        else: raise LookupError(f"VTRWin: optn={self.optn} not implemented")

        return self.low + (1 - self.low) * y

    def flin(self, x):
        """
        Linear window over normalized time.

        Args:
            x (Tensor): Normalized time.

        Returns:
            Tensor: Window value.
        """

        y = torch.pow(1 - x, self.pinv)
        return y

    def frcos(self, x):
        """
        Raised cosine window over normalized time.

        Args:
            x (Tensor): Normalized time.

        Returns:
            Tensor: Window value.
        """
         
        y = torch.cos(x * 0.5 * torch.pi).square().pow(self.pinv)

        return y


    def qi(self, x):
        """Affine map from [0,1] → [-1,1]."""
        return 2 * x - 1

    def q0(self, x, m=0.5):
        """
        Peak-shifting transform.

        Args:
            x (Tensor): Input in [0,1].
            m (float): Peak location.

        Returns:
            Tensor: Symmetric magnitude centered at shifted peak.
        """
        x, m = self.qi(x), self.qi(m)
        x -= m
        x /= (1 - m * torch.sign(x))
        return torch.abs(x)

    def q1(self, x, e=0):
        """
        Flat-top shaping transform.

        Args:
            x (Tensor): Input.
            e (float): Flat-top width.

        Returns:
            Tensor: Thresholded and rescaled output.
        """
        return torch.clamp_min((x - e) / (1 - e), min=0)

    def tfx(self, x, m, e):
        """
        Composite transform over normalized time.

        Applies peak shifting followed by flat-top shaping.

        Args:
            x (Tensor): Normalized time.
            m (float): Peak location.
            e (float): Flat-top width.

        Returns:
            Tensor: Transformed coordinate.
        """
        return self.q1(self.q0(x, m), e)


# =========================================================
# Statistical estimators for the Expectation fcn.
# =========================================================
class stat():
    @staticmethod
    def ema(u, x, rho):
        """
        Recursive estimator: EMA

        The EMA mode is a special operating case of a single-pole lowpass filter.

        Args:
            u (Tensor): input sample.
            x (Tensor): prior estimate of the true mean.
            rho (float): confidence weight on `x` in (0, 1).
            t (int): iteration index.

        Returns:
            (Tensor, Tensor): output, updated state.
        """
        x = rho * x + (1 - rho) * u
        return 1*x, x
    
    
    @staticmethod
    def cema(u, x, rho, t):
        """
        Recursive estimator: Corrects transient estimation bias of `utl.ema`.

        Here `x` starts from a possibly poor prior estimate of `E{x}`.

        This prior estimate is then recursively updated.
        """
        y, x = stat.ema(u, x, rho)
        y /= utl.ompow(rho, t)
        return y, x    
    
    @staticmethod
    def lse(u, x, rho, t):
        """
        Linear shrinkage estimator.

        Here `x` starts from a possibly strong prior estimate of `E{x}`.
        
        Robust, memory-efficient estimator: fixed prior, x
        
        Args:
            u (Tensor): input sample.
            x (Tensor): a possibly true estimate of the mean.
            rho (float): confidence weight on `x` in (0, 1).
            t (int): iteration index.
        """
        x *= rho
        x += (1 - rho) * u
        x /= utl.ompow(rho, t)
        return 1*x, x


# =========================================================
# Matrix p-th root estimators.
# =========================================================
class msr():

    @staticmethod
    def hns(M, eps=1e-8, K=8, inv=True):
        """Higham's Newton-Schulz (NS) coupled iteration for the Matrix Square Root and its inverse. `hns` is equivalent to `slp2`.

        Quadratic order of convergence
        Cost: 3 mat-mul, 3n^3 flops per iteration + 3 O(1).

        Args:   
            M (Tensor): input square matrix
            eps (float, optional): regularization constant. Defaults to 1e-8.
            K (int, optional): number of iterations. Defaults to 5.  
            inv (bool, optional): if True, compute M^{-1/2}, else M^{1/2}. Defaults to True.

        Refs:
            Higham, N. J. (2008). Functions of Matrices. Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898717778
            [2017] https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
            [2020] https://github.com/photosynthesis-team/piq/issues/190
        """
        I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) 
        M = M + eps*I
        trace = torch.linalg.norm(M).clip(min=eps)  
        c = 1/trace
        M *= c
        Z, I3 = 1*I, 3*I
        for _ in range(K): 
            # R = I - Z @ M
            # T = R + (0.5 * (I + (Z @ M)))
            # Z = (1.5 * Z) - 0.5*(Z @ (M @ Z) @ Z)
            T = 0.5 * (I3 - (Z @ M))
            M, Z = M @ T, T @ Z 
            # if torch.linalg.norm(R) < 1e-5: break

        if inv: return torch.sqrt(c)*Z
        else: return torch.sqrt(trace)*M

    @staticmethod
    def slp2(M, eps=1e-8, K=8):
        """Slobodan Lakic's (SL) 2nd-order coupled iteration for the Inverse Square Root of Matrix `M`.

        coefficient vector b = [3, -1]/2

        Quadratic order of convergence
        Cost: 3 mat-mul, 3n^3 flops per iteration + 3 O(1).

        Args:   
            M (Tensor): input square matrix
            eps (float, optional): regularization constant. Defaults to 1e-8.
            K (int, optional): number of iterations. Defaults to 5.  

        Refs:
            Lakić, S. (1998).
            On the computation of the matrix k-th root.
            Zeitschrift für Angewandte Mathematik und Mechanik, 78(3), 167-168.
            https://doi.org/10.1002/(SICI)1521-4001(199803)78:3<167::AID-ZAMM167>3.0.CO;2-R
        """
        I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) 
        S = M + eps*I
        trace = torch.linalg.norm(M).clip(min=eps)  
        c = 1/trace
        S *= c
        Z, I3 = 1*I, 3*I
        for _ in range(K): 
            T = 0.5 * (I3 - S)
            Z, S = Z @ T, T @ T @ S 

        return torch.sqrt(c)*Z


    @staticmethod
    def slp4(M, eps=1e-8, K=4):
        """Slobodan Lakic's (SL) 4th-order coupled iteration for the Inverse Square Root of Matrix `M`.

        coefficient vector b = [35, -35, 21, -5]/16
        
        Quartic order of convergence
        Cost: 5 mat-mul, 5n^3 flops per iteration + 6 O(1).
        
        Args:   
            M (Tensor): input square matrix
            eps (float, optional): regularization constant. Defaults to 1e-8.
            K (int, optional): number of iterations. Defaults to >= 4.  

        Refs:
            Lakić, S. (1998).
            On the computation of the matrix k-th root.
            Zeitschrift für Angewandte Mathematik und Mechanik, 78(3), 167-168.
            [https://doi.org/10.1002/(SICI)1521-4001(199803)78:3<167::AID-ZAMM167>3.0.CO;2-R]
        """
        I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype) 
        S = M + eps*I
        trace = torch.linalg.norm(M).clip(min=eps)  
        c = 1/trace
        S *= c
        Z, I21 = 1*I, 1.3125*I
        for _ in range(K): 
            T = (2.1875 * (I - S)) + ((I21 - (0.3125*S)) @ S @ S)
            Z, S = Z @ T,  S @ T @ T
        return torch.sqrt(c)*Z

    @staticmethod
    def mat_s2norm(g, eps):
        """ returns schatten 2-normalized matrix, g
        """
        # covariance measure of matrix g
        s2norm = msr.slp4(g @ g.T, eps, 4)
        # return normalized matrix.
        return s2norm @ g

# =========================================================
# state struct for model parameters
# =========================================================
class PState:
    """
    Per-parameter state for the SGM optimizer.

    Attributes:
        q (Tensor): Gradient filter state.
        gpow (Tensor): Second-moment estimate.
        gkurt (Tensor): Fourth-moment estimate.
        cwin (VTRWin): Window function.
        max_mu (Tensor): Maximum allowwable (uniform) step-size bound.
    """

    __slots__ = ("w", "q", "gpow", "gfrt",
        "cwin", "sn", "max_mu", "gprops", "lowdim",  
        "rho", "eps", "wcf", "lse", "lpf", "beta", "coeff",
        "update_fn")

    def __init__(self, group, p, gprops, pmom):

        # unpack once
        self.rho, self.eps, self.wcf, self.lse = group["stat_cfg"]

        # init. fix for: tiny params., and tiny param. norm
        masks = (p.abs() < self.eps, p.norm() < self.eps)
        p[masks[0]] += self.eps * p[masks[0]].sign()
        p[masks[1]] /= p[masks[1]].norm() + self.eps

        # self.w = p.real.clone()
        self.gprops = gprops

        # unpack
        optn, m, e, l, T = group["win_cfg"]
        vtrw_cfgs = (optn, pmom, m, e, l, T)
        self.cwin = VTRWin(*vtrw_cfgs)

        # Canonical filter implementation index:
        #     0 → (zero-parameterized direct form II)
        #     1 → (transposed form)
        #     2 → (gap-parameterized direct form II)
        #     3 → (transposed gap form)
        #     4 → (gain-parameterized direct form II)
        #     5 → (transposed gain form)
        # - γ (zero location) if ftype ∈ {0,1}
        # - δ (pole-zero gap) if ftype ∈ {2,3}

        FOFILTER_CONFIGS = [
            ("df2",  "zero"),  # 0            
            ("df2t", "zero"),  # 1
            ("df2",  "gap"),   # 2
            ("df2t", "gap"),   # 3
            ("df2", "gain"),   # 4
            ("df2t", "gain"),  # 5
        ]
        lpf_cfgs = FOFILTER_CONFIGS[1]
        canon, param = lpf_cfgs

        self.max_mu = torch.tensor(group["tr_cfg"][0], 
                    dtype=p.dtype, device=p.device)
        
        self.beta, self.coeff = get_zero(*group["tr_cfg"][1:-1])

        if gprops[0] is False:
            self.q = torch.zeros_like(p)
            self.gpow = torch.zeros_like(p)
            if not self.lse:
                self.gfrt = torch.ones_like(p)
            self.lpf = FirstOrderLPF(canon=canon, param=param)

        else:
            p = p.real.clone()
            if p.ndim > 1:
                if gprops[1]: p = p.reshape(p.shape[0], -1)
                self.lowdim = p.shape[0] > p.shape[1]
                # smaller dim. first
                if self.lowdim: p = p.T
            else:
                p = p.reshape(-1, 1)
                self.lowdim = False     

            self.sn = msr.mat_s2norm

            self.q = torch.zeros_like(p)
            self.gpow = torch.zeros_like(p)
            if not self.lse:
                self.gfrt = torch.ones_like(p)
            self.lpf = FirstOrderLPF(canon=canon, param=param, mode="core")

    
        self.update_fn = None

    def build_update_fn(self, parent):
        """
        Build a branch-free update kernel specialized
        for this parameter group (vector or matrix).
        """
        if parent.p == -2:
            return self._build_adam_or_rmsprop_update(parent)
        else:
            if not self.gprops[0]:                 
                if not self.lse: return self._build_vector_update(parent)
                else: return self._build_vector_lse_update(parent)
            else:
                if not self.lse: return self._build_matrix_update(parent)
                else: return self._build_matrix_lse_update(parent)
        
    def _build_vector_update(self, parent):

        rho, eps, wcf = self.rho, self.eps, self.wcf
        base_mu, beta, coeff  = self.max_mu, self.beta, self.coeff

        # etas = self.lpf.gains_io(beta, coeff, "zero")
        # ceta = 1 - etas[0]

        def update(p, grad, t):

            # ------------------------------
            # regularize gradient (lowpass filter)
            # ------------------------------
            q = self.q
            v, q = self.lpf.step(grad, q, beta, coeff, t)
            self.q = q

            # ----------------------------------------------
            # learning rate fcn = mu_t / [E{g^p}^(1/p)],
            # ----------------------------------------------
            
            # --- (vanishing) trust-region shaping of update step
            mu =  base_mu * self.cwin.step(t)

            # --- enforce p-moment bound: E{|step|^p} <= base_mu^p

            ## - base scale: second moment estimate
            gpow = self.gpow
            m2, gpow = stat.cema(grad*grad, gpow, rho, t)
            self.gpow = gpow

            # - base normalization
            m2 = torch.sqrt(m2)
            v /= (m2 + eps)

            ## - normalized p-th moment estimate   
            gfrt = self.gfrt
            mp, gfrt = stat.cema(torch.abs(v).pow(parent.p), gfrt, rho, t)
            self.gfrt = gfrt 
            # _pprt(t, 'mp', mp.mean())
            mp.clip_(min=1).pow_(parent.pinv).clip_(min=1)
            # - p-moment normalization
            v /= (mp + eps)
            
            # --- [regularized] update step (weight-decay)
            v += wcf * p
            v *= -mu
            p += v

        return update

    def _build_vector_lse_update(self, parent):

        rho, eps, wcf = self.rho, self.eps, self.wcf
        base_mu, beta, coeff  = self.max_mu, self.beta, self.coeff

        # etas = self.lpf.gains_io(beta, coeff, "zero")
        # ceta = 1 - etas[0]

        def update(p, grad, t):

            # ------------------------------
            # regularize gradient (lowpass filter)
            # ------------------------------
            q = self.q
            v, q = self.lpf.step(grad, q, beta, coeff, t)
            self.q = q

            # ----------------------------------------------
            # learning rate fcn = mu_t / [E{g^p}^(1/p)],
            # ----------------------------------------------
            
            # --- (vanishing) trust-region shaping of update step
            mu =  base_mu * self.cwin.step(t)

            # --- enforce p-moment bound: E{|step|^p} <= base_mu^p

            ## - base scale: second moment estimate
            gpow = self.gpow
            m2, gpow = stat.cema(grad*grad, gpow, rho, t)
            self.gpow = gpow
            # - base normalization
            m2 = torch.sqrt(m2)
            v /= (m2 + eps)

            ## - normalized p-th moment estimate   
            mp, _ = stat.lse(v.abs().pow(parent.p), 1, rho, t)
            # _pprt(t, 'mp', mp.mean())
            mp.clip_(min=1).pow_(parent.pinv).clip_(min=1)
            # - p-moment normalization
            v /= (mp + eps)
            
            # --- [regularized] update step (weight-decay)
            v += wcf * p
            v *= -mu
            p += v

        return update

    def _build_matrix_update(self, parent):

        rho, eps, wcf   = self.rho, self.eps, self.wcf
        base_mu, beta, coeff  = self.max_mu, self.beta, self.coeff

        def update(p, grad, t):

            # original -> reshaped
            if grad.ndim > 1:
                if self.gprops[1]:
                    grad = grad.reshape(grad.shape[0], -1)
                if self.lowdim:
                    grad = grad.T
            else:
                grad = grad.reshape(-1, 1)

            q = self.q
            v, q = self.lpf.step(grad, q, beta, coeff, t)
            self.q = q      

            # ------------------------------
            # learning rate function
            # ------------------------------
            mu = base_mu * self.cwin.step(t)

            # --- enforce p-moment bound: E{|step[i,j]|^p} <= base_mu^p

            ## - base scale: second moment
            gpow = self.gpow
            m2, gpow = stat.cema(grad*grad, gpow, rho, t)
            self.gpow = gpow

            ## - base normalization
            m2 = torch.sqrt(m2)
            v /= (m2 + eps)

            ## - normalized p-th moment estimate   
            gfrt = self.gfrt
            mp, gfrt = stat.cema(v.abs().pow(parent.p), gfrt, rho, t)
            self.gfrt = gfrt
            
            mp.clip_(min=1).pow_(parent.pinv)
            # - p-moment normalization
            v /= (mp + eps)

            # --- enforce spectral-norm bound on update step matrix
            v = self.sn(v, eps)

            if self.lowdim: v = v.T
            if self.gprops[1]: v = v.reshape(self.gprops[-1])

            v += wcf * p
            v *= -mu
            p += v

        return update

    def _build_matrix_lse_update(self, parent):

        rho, eps, wcf   = self.rho, self.eps, self.wcf
        base_mu, beta, coeff  = self.max_mu, self.beta, self.coeff

        def update(p, grad, t):

            # original -> reshaped
            if grad.ndim > 1:
                if self.gprops[1]:
                    grad = grad.reshape(grad.shape[0], -1)
                if self.lowdim:
                    grad = grad.T
            else:
                grad = grad.reshape(-1, 1)

            q = self.q
            v, q = self.lpf.step(grad, q, beta, coeff, t)
            self.q = q      

            # ------------------------------
            # learning rate function
            # ------------------------------
            mu = base_mu * self.cwin.step(t)

            # --- enforce p-moment bound: E{|step[i,j]|^p} <= base_mu^p

            ## - base scale: second moment
            gpow = self.gpow
            m2, gpow = stat.cema(grad*grad, gpow, rho, t)
            self.gpow = gpow

            ## - base normalization
            m2 = torch.sqrt(m2)
            v /= (m2 + eps)

            ## - normalized p-th moment estimate   
            mp, _ = stat.lse(v.abs().pow(parent.p), 1, rho, t)
            mp.clip_(min=1).pow_(parent.pinv)
            # - p-moment normalization
            v /= (mp + eps)

            # --- enforce spectral-norm bound on update step matrix
            v = self.sn(v, eps)

            if self.lowdim: v = v.T
            if self.gprops[1]: v = v.reshape(self.gprops[-1])

            v += wcf * p
            v *= -mu
            p += v

        return update

    def _build_adam_or_rmsprop_update(self, parent):

        rho, eps, wcf = self.rho, self.eps, self.wcf
        base_mu, beta, coeff  = self.max_mu, self.beta, self.coeff

        def update(p, grad, t):

            # ------------------------------
            # regularize gradient (lowpass filter)
            # ------------------------------
            q = self.q
            v, q = self.lpf.step(grad, q, beta, coeff, t)
            self.q = q

            # --- (vanishing) trust-region shaping of update step
            mu =  base_mu * self.cwin.step(t)

            # --- base scale: second moment estimate
            gpow = self.gpow
            m2, gpow = stat.cema(grad*grad, gpow, rho, t)
            self.gpow = gpow

            m2 = torch.sqrt(m2)
            v /= (m2 + eps)
            
            # --- [regularized] update step (weight-decay)
            v += wcf * p
            v *= -mu
            p += v

        return update
    
# =========================================================
# SGM optimizer
# =========================================================
class SGM_GMAKE:
    """
    
    ### Stochastic Gradient Method with Gmake
    
    * Gmake (Gradient Moment and Kurtosis Estimation)    
    * Gmake is a p-th moment trust-region constrained update step-size of the stochastic gradient method, where p >= 1.

    The Gmake framework seeks to understand the algorithimic components in Adam from a trust-region perspective.

    Parameter updates combine:
        • iterative p-th order moment-based normalization (second moment + normalized p-th moment)   

        • [optional] trust-region preserving first-order filtering of the input gradient (generalized momentum family)

        • [optional] variational trust-region window shaping (learning-rate schedule family)

        • [optional] spectral-norm trust-region constrained step sizes
         
    Several aspects of training techniques studied separately can be understood from a trust-region perspective.

    ---------------------------------------------------------------------
    Key Configurations
    ---------------------------------------------------------------------

    params : iterable
        Iterable of parameters or parameter groups (PyTorch-style).

    p : float, [1, 4], default=4
        moment order for trust-region constraint:

    tr_cfg : (max_mu, beta, gamma, use_mat), default=(0.001, 0.9, 0.5528)
        Configuration for elementwise (vector) updates:

            max_mu, μ : float
                Maximum allowable step size of the vectorized parameter update.

            beta, β : float
                First-order filter pole 0 ≤ β < 1 (controls smoothing / momentum).

            gamma, γ : float | str
                Filter zero parameter γ < β (controls smoothing / momentum):
                If 'str', pre-defined options include
                gamma = "off":  γ = β = 0 → plain SGD
                gamma = "phb":  γ = 0 → Heavy-ball momentum
                gamma = "nag":  γ = β / (1 + β) → Nesterov momentum
                gamma = "vrg":  γ = max(-β, 1-sqrt{2*(1-β)}) → efficient variance reduction per unit gain design tradeoff.
                
            use_mat : bool
                matrix view of the update step. Enable or disable spectral norm constrained step-size.

    stat_cfg : (rho, eps, wcf, lse), default=(0.999, 1e-10, 0, False)
        Statistical estimation and regularization:

            rho : float
                statistical mean estimation coefficient. 
                Used in exponential moving average estimators and linear shrinkage estimators.

            eps : float
                Tiny numerical stability constant.

            wcf : float
                Weight-decay coefficient.

            lse : bool
              Enable using a shrinkage estimator instead of an exponential moving average for the normalized p-th moment estimate.  


    win_cfg : (optn, m, e, l, T), default=(1, 0, 0, 0, 1e16)
        Trust-region aware step-size schedule: controls variation of the update step-size over the normalized time window [0,1].

            optn :
                0 → constant, no schedule enabled
                1 → linear schedule envelope
                2 → raised-cosine envelope

            m : float
                Peak location in [0,1).

            e : float
                Flat-top region of width in [0,1) enforcing a plateau
                where the window remains near its maximum value.

            l : float
                user-configured minimum normalized window value: l in [0,1).

            T : int
                Window period and therefore the time-scale
                over which modulation occurs.            
                - T = total training iterations to produces a single window (no cycling)
                - T ≪ total iterations to induce periodic modulation (cyclical learning behavior)
                - T can be set to iterations per epoch to induce cyclical learning behavior if epoch > 1.

    ---------------------------------------------------------------------
    Internal State (Per Parameter)
    ---------------------------------------------------------------------

    Each parameter is associated with a PState object containing:

        q : Tensor
            Internal filter state (recursive gradient accumulator)

        gpow : Tensor
            Running estimate of second moment E[g²]

        cwin : VTRWin
            Time-varying window controlling trust-region scaling

        max_mu : Tensor
            Maximum allowed step magnitude

        lpf : list[callable]
            Set of filter implementations

        gprops : tuple
            Shape classification:
                (is_matrix_mode, reshaped_flag, original_shape)

        lowdim : bool
            Controls dimension orientation in matrix mode

        sn : callable
            Matrix normalization operator based on inverse square-root

            
    ---------------------------------------------------------------------
    Design Notes
    ---------------------------------------------------------------------   

    The parameter update steps satisfy E[|step|^p] ≤ max_mu^p

    1. apply first-order filter (momentum): v_t = FILTER(grad)

    2. apply learning rate:
        - 2.1. tapering window (learning-rate schedule)
        μ_t = max_mu × window(t)

        - 2.2. p-moment normalization (learning-rate mechanism)
        step = v_t / (E[|v_t|^p])^(1/p) 

    3. apply weight-decay:
        - step += weight_decay term
        - step *= -μ_t

    4. update: p ← p + step     


    • A matrix-view of the update step process is enabled if `use_mat` is set as `True` in `tr_cfg`:
        Parameters are automatically classified into vector or matrix modes.
        Applies additional spectral-norm normalization via an iterative inverse square-root estimation of the step covariance matrix (step × step').
        -2.3. step = inv_mat_sqrt(step)

    ---------------------------------------------------------------------
    Example
    ---------------------------------------------------------------------
        
    ```python
    model = net_model()
    num_iters = int(1e9)

    optimizer = SGM_GMAKE(
        model.parameters(), p=2,
        tr_cfg=(1e-3, 0.9, 0.55, False),
        stat_cfg=(0.999, 1e-10, 1e-4, False),
        win_cfg=(2, 0.5, 0.1, 0, num_iters)
    )

    for x, y in dataloader:
        optimizer.zero_grad()

        loss = model(x).loss(y)
        loss.backward()

        optimizer.step()
    ```
        
    """

    @torch.no_grad()
    def __init__(self, params, *, p=2,
                 tr_cfg=(0.001, 0.9, 'vrg', False), 
                 stat_cfg=(0.999, 1e-10, 0, False),
                 win_cfg=(1, 0, 0, 0, 1e16),
                ):

        self.param_groups = []
        self.state = {}

        if p == -2:
            # use to build adam baseline,
            # so it benefits from this interface.
            self.p = p
        else:
            if (p < 0 or p > 4): 
                raise ValueError(f"Require '0 <= p <= 4', not p={p}.")  
            self.p = p
            self.pinv = 1/p

        # note: after this p stands for model parameter

        if isinstance(params, dict):
            params = [params]
        elif not isinstance(params, list):
            params = [{"params": params}]

        for group in params:
            gparams = list(group["params"])

            # mapping: key -> default tuple
            param_defaults = {                
                "tr_cfg": tr_cfg,   
                "win_cfg": win_cfg,
                "stat_cfg": stat_cfg, 
            }
            # ensure all keys exist
            for key, default in param_defaults.items():
                group[key] = group.get(key, default)

            dtype, device = None, None
            for p in gparams:
                if p.requires_grad is False: continue

                dtype, device = p.dtype, p.device
                gprops = utl.classdim(p, usemat=group["tr_cfg"][-1])
                self.state[p] = PState(group, p, gprops, self.p)
                self.state[p].update_fn = self.state[p].build_update_fn(self)

            if dtype is None: continue

            group["t"] = torch.tensor(0.0, dtype=dtype, device=device)
            group["params"] = gparams
            self.param_groups.append(group)

    @torch.no_grad()
    def step(self):
        """
        Perform one optimization step.
        """
        for grp in self.param_groups:

            t = grp["t"] + 1
            grp["t"] = t
  
            for p in grp["params"]:
                grad = p.grad
                if grad is None: continue

                st = self.state[p]
                st.update_fn(p, grad, t)


    def zero_grad(self):
        '''Clear gradient values in memory for all trainable parameters.'''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    # ================================================
    # Named presets (user-facing API)
    # ================================================

    @classmethod
    def Adam(cls, params,
                tr_cfg=(0.001, 0.9, 'phb'), 
                stat_cfg=(0.999, 1e-10, 0),
                win_cfg=(0, 0, 0, 0, 1e32),):
        
        return SGM_GMAKE(params, p=-2, 
                tr_cfg=(*tr_cfg, False), 
                stat_cfg=(*stat_cfg, False), 
                win_cfg=win_cfg)

    @classmethod
    def RMSProp(cls, params,
                tr_cfg=(0.001,), 
                stat_cfg=(0.999, 1e-10, 0),
                win_cfg=(0, 0, 0, 0, 1e32),):
        
        return SGM_GMAKE(params, p=-2, 
                tr_cfg=(*tr_cfg, 0, 'phb', False), 
                stat_cfg=(*stat_cfg, False), 
                win_cfg=win_cfg)


