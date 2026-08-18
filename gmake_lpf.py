import torch
import math

# ------------------------------------------------------------------
# Filter Coefficient Rules (Internal)
# ------------------------------------------------------------------

# Let β = beta.

# For ftype ∈ {0,1}:  coeff = γ
# For ftype ∈ {2,3}:  coeff = δ

# Mapping:

#     filter = "none":
#         γ = β, δ = 0

#     filter = "hb":
#         γ = 0, δ = β

#     filter = "nag":
#         γ = β / (1 + β)
#         δ = β² / (1 + β)

def get_zero(beta, coeff):
    if isinstance(coeff, float): return beta, coeff
    elif isinstance(coeff, str):
        if coeff == "off": return 0, 0
        elif coeff == "phb": 
            zero = 0
        elif coeff == "nag":
            zero = round(beta / (1 + beta), 4)
        elif coeff == "vrg":
            zero = round(max(-beta, 1 - math.sqrt(2*(1-beta))), 4)
        else:
            raise ValueError(f"unknown preset, '{coeff}'!")
        return beta, zero

class FirstOrderLPF:
    """
    First-order IIR low-pass filter with compiled step kernels.

    IIR means infinite impulse response.


    The filter implements:

        H(z) = eta * (1 - gamma z^-1) / (1 - beta z^-1)

    where transient bias correction and normalization are optionally applied.

    During initialization, each configuration: 
    (canonical structure × filter param type × normalization mode) 
    is generated as a specialized step function with no runtime branching.

    ----------------------------------------------------------------------
    Configuration
    ----------------------------------------------------------------------

    structure : {"df2", "df2t"}
        Defines the state-space realization:
            - df2  : Direct Form II
            - df2t : Transposed Direct Form II

    param : {"zero", "gap"}
        Parameterization of numerator:
            - zero : explicit zero location (gamma)
            - gap  : pole-zero gap (delta = beta - gamma)

    mode : {"core", "cheap", "full"}
        Controls normalization and transient correction:
        
            core  :
                No input/output normalization.
                No bias correction.
                Raw system Hc(z) = (1 - beta z^{-1}) / (1 - gamma z^{-1}).

            cheap :
                Applies:
                    - input scaling (1 - beta)
                    - output scaling (1 - gamma)
                    - pole-only bias correction

            full :
                Applies:
                    - full I/O normalization
                    - exact transient correction:
                        (1 - gamma^t) / (1 - beta^t)

    ----------------------------------------------------------------------
    Step Function
    ----------------------------------------------------------------------

    The generated step function has signature:

        v, q = step(x, q, beta, coeff, t)

    Args:
        x     : input sample (float or Tensor)
        q     : scalar internal state
        beta  : pole coefficient (|beta| < 1)
        coeff :
            - gamma if param="zero"
            - delta if param="gap"
        t     : timestep (integer, t >= 1)

    Returns:
        v : filtered output
        q : updated state

    ----------------------------------------------------------------------
    Notes
    ----------------------------------------------------------------------

    - No branching occurs inside `step()`
    - All normalization constants are computed per-step (cheap scalar ops)
    - gamma is recovered automatically when using gap parameterization
    - Suitable for high-performance loops and differentiable pipelines

    ----------------------------------------------------------------------
    Numerical Properties

    - Stability: |beta| < 1
    - Time complexity: O(1) per update
    - Memory: single scalar state
    - All configurations implement identical transfer functions

    ----------------------------------------------------------------------

    ....
    
    ----------------------------------------------------------------------
     Parameter relationships
    ----------------------------------------------------------------------

        beta  : pole coefficient
        gamma : zero coefficient
        delta : pole-zero gap, delta = beta - gamma

        etai  : input signal normalization factor
                etai = 1 - beta

        etao  : output signal normalization constant
                etao = 1 - gamma

        eta   : DC gain normalization constant
                eta = etai / etao

        etao = etai + delta

        Normalization is performed as:
            - g_t = etai * grad_t   (input signal normalizaion)
            - v_t = Hc{g_t} / etao  (output signal normalization)

    Notes:
        - etai rescales the input signal before filtering
        - etao rescales the output after filtering
        - eta ensures unit DC gain
        - The unnormalized configuration corresponds assumes the normalization  
          from the input/output scaling has been externally applied or absorbed in an external learning rate. 

    ----------------------------------------------------------------------
    Transient Bias Correction
    ----------------------------------------------------------------------

    The bias correction corrects for slow convergence to a constant step input. During early itererations, the output of the filter is typically biased.

    Two correction modes are implemented:

    1) FULL scaling:
        Corrects both pole and zero transient effects:

            v_t <- v_t * (1 - gamma^t) / (1 - beta^t)

        This fully compensates the transient of the transfer function
        H(z), and yields an exact unbiased response assuming exact parametrization.

    2) CHEAP scaling:
        Corrects only the dominant pole transient:

            v_t <- v_t *  (etao if t == 1 else 1) / (1 - beta^t)

        This ignores numerator (zero) transient effects.

        In practice this is usually sufficient when:
            - the pole > zero [typical case]
            - the dominant bias arises from the pole (state accumulation)

        The cheap form is computationally simpler and often indistinguishable for most practical settings.

    ---------------------------------------------------------------------
    Impact and Role of Bias Correction
    ----------------------------------------------------------------------

    The bias correction terms do NOT correct initialization bias per se.
    Instead, they compensate for finite-time accumulation in the filter's
    recursive structure.

    The filter involves exponentially weighted sums of the form:

        Σ beta^k  and  Σ gamma^k

    which, for finite t, evaluate to:

        (1 - beta^t) / (1 - beta)
        (1 - gamma^t) / (1 - gamma)

    rather than their infinite-horizon limits.

    As a result, the raw filter output corresponds to a *truncated*
    exponential accumulation, not the steady-state transfer function.

    Bias correction rescales the output to account for the missing
    tail of these geometric series.

    CHEAP scaling:
        - Compensates for finite-time accumulation of the pole (beta)
        - Corrects normalization of the dominant geometric sum
        - Does not account for numerator (zero-related) truncation

    FULL scaling:
        - Compensates both:
            * denominator accumulation (beta)
            * numerator accumulation (gamma)
        - Aligns finite-time behavior with the full transfer function


    Interpretation
    - The correction terms arise from exact identities of geometric sums,
      not from heuristic bias removal.

    - The discrepancy being corrected is:

          finite sum  ≠  infinite sum

      rather than "biased estimation due to initialization".

    - The pole (beta) governs accumulation of past state contributions,
      hence dominates scaling behavior.

    - The zero (gamma) affects the numerator structure and introduces
      a secondary correction term.

    Practical Effect
    - Without correction:
        Output reflects truncated exponential accumulation

    - Cheap correction:
        Restores correct normalization of the dominant accumulation term

    - Full correction:
        Restores full finite-time equivalence to the intended transfer
        function decomposition


    These corrections do not alter the underlying system dynamics or
    steady-state behavior; they only rescale early transient responses.

    """

    def __init__(self, canon="df2", param="zero", mode="cheap"):

        self.step = self._build_step(canon, param, mode)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def ompow(rho, t):
        return 1 - rho**t

    @staticmethod
    def gains_io(beta, coeff, param):
        etai = 1 - beta
        if param == "zero":
            gamma = coeff
            etao = 1 - gamma
        else:
            delta = coeff
            etao = etai + delta
        eta = etai / etao
        return eta, etai, etao, coeff

    # ------------------------------------------------------------------
    # Kernel builder (key idea)
    # ------------------------------------------------------------------

    def _build_step(self, structure, param, mode):
        """
        Generate a specialized step function.

        Result: No branching inside the time loop.
        """

        

        # --------------------------------------------------------------
        # Resolve parameter roles
        # --------------------------------------------------------------
        if param == "zero":
            def iog(beta, coeff, tn):
                etai = 1 - beta
                gamma = coeff
                etao = 1 - gamma
                return etai, etao, coeff
            
        elif param == "gap":
            def iog(beta, coeff, tn):
                etai = 1 - beta
                delta = coeff
                etao = etai + delta
                return etai, etao, coeff
            

        # --------------------------------------------------------------
        # Choose core recurrence
        # --------------------------------------------------------------

        if structure == "df2" and param == "zero":

            def gamma(beta, gamma): 
                return gamma

            def core(g, q, beta, gamma):
                u = beta * q + g
                v = u - gamma * q
                return v, u

        elif structure == "df2" and param == "gap":

            def gamma(beta, delta): 
                return beta - delta

            def core(g, q, beta, delta):
                v = delta * q + g
                q = beta * q + g
                return v, q

        elif structure == "df2" and param == "gain":

            def core(g, q, beta, trust):
                eta = 1 - trust
                etai = 1 - beta
                v = eta*g + trust*q 
                q = beta*q + etai*g
                gamma = (beta-trust)/eta
                return v, q, gamma
            
        elif structure == "df2t" and param == "zero":

            def gamma(beta, gamma): 
                return gamma
            
            def core(g, q, beta, gamma):
                v = q + g
                q = beta * v - gamma * g
                return v, q
            
        elif structure == "df2t" and param == "gap":

            def gamma(beta, delta): 
                return beta - delta
            
            def core(g, q, beta, delta):
                v = q + g
                q = beta * q + delta * g
                return v, q
            
        elif structure == "df2t" and param == "gain":

            def core(g, q, beta, trust):
                eta = 1 - trust
                etam = beta - trust
                v = eta*g + q 
                q = beta*v - etam*g
                gamma = etam/eta
                return v, q, gamma
            
        else:
            raise NotImplementedError(f"You passed: '{structure}', and '{param}'. This is not a valid combination !")


        # pre-bind scaling behavior
        if mode == "full":

            def correct_bias(v, beta, gamma, t):
                return v * (
                    self.ompow(gamma, t) / self.ompow(beta, t)
                )

        elif mode == "cheap":

            def correct_bias(v, beta, gamma, t):
                return v * (
                    (1 - gamma if t == 1 else 1) / self.ompow(beta, t)
                )
        # --------------------------------------------------------------
        # Build final step function
        # --------------------------------------------------------------

        if mode == "core":
            if param in ["zero", "gap"]:
                def step(x, q, beta, coeff, t, tn=0):
                    v, q = core(x, q, beta, coeff)
                    return v, q
            elif param in ["gain"]:
                def step(x, q, beta, coeff, t, tn=0):
                    v, q, _ = core(x, q, beta, coeff)
                    return v, q           
        else:
            if param in ["zero", "gap"]:
                def step(x, q, beta, coeff, t, tn=0):
                    etai, etao, coeff = iog(beta, coeff, tn)
                    g = etai*x
                    v, q = core(g, q, beta, coeff)
                    v = v/etao
                    v = correct_bias(v, beta, gamma(beta, coeff), t)
                    return v, q
            else:
                def step(x, q, beta, coeff, t, tn=0):
                    v, q, gamm = core(x, q, beta, coeff)
                    v = correct_bias(v, beta, gamm, t)
                    return v, q             

        return step



# =============================================================================
# FREQUENCY RESPONSE 
# =============================================================================

def compare_frequency_response(beta=0.9, gamma=0.33):
    """
    Unified frequency response comparison.

    Combines:
        - normalized vs unnormalized
        - magnitude + phase
        - single figure, aligned axes

    Notes:
        - Scaling (cheap/full) does NOT affect frequency response
        - Only I/O normalization (etai/etao) changes magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt

    w = np.linspace(0, np.pi, 512)
    zinv = np.exp(-1j * w)

    # Unnormalized
    H_raw = (1 - gamma * zinv) / (1 - beta * zinv)

    # Normalized
    etai = 1 - beta
    etao = 1 - gamma
    H_norm = (etai / etao) * H_raw

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # ---- Magnitude
    axs[0, 0].plot(w, np.abs(H_raw), label="unnormalized")
    axs[0, 0].set_title("Magnitude (Raw)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(w, np.abs(H_norm), label="normalized", color="tab:orange")
    axs[0, 1].set_title("Magnitude (Normalized)")
    axs[0, 1].grid(True)

    # ---- Phase
    axs[1, 0].plot(w, np.angle(H_raw))
    axs[1, 0].set_title("Phase (Raw)")
    axs[1, 0].set_xlabel("Frequency (rad/sample)")
    axs[1, 0].grid(True)

    axs[1, 1].plot(w, np.angle(H_norm), color="tab:orange")
    axs[1, 1].set_title("Phase (Normalized)")
    axs[1, 1].set_xlabel("Frequency (rad/sample)")
    axs[1, 1].grid(True)

    fig.suptitle("First-Order LPF Frequency Response Comparison")
    plt.tight_layout()
    plt.show()

# =============================================================================
# STEP RESPONSE COMPARISON
# =============================================================================

def compare_step_responses():
    """
    Step response comparison across:
        - Structures: df2, df2t
        - Parameterizations: zero, gap
        - Modes: core, cheap, full

    Demonstrates:
        - Equivalence across realizations
        - Effect of normalization and bias correction
    """

    import matplotlib.pyplot as plt

    T = 120
    x = torch.ones(T)

    beta = 0.9
    gamma = 0.2
    delta = beta - gamma

    configs = [
        ("df2",  "zero"),
        ("df2",  "gap"),
        ("df2t", "zero"),
        ("df2t", "gap"),
    ]

    modes = ["core", "cheap", "full"]

    def run(structure, param, mode):
        f = FirstOrderLPF(structure, param, mode)

        q = torch.tensor(0.0)
        y = []

        coeff = gamma if param == "zero" else delta

        for t in range(1, T + 1):
            v, q = f.step(x[t-1], q, beta, coeff, t)
            y.append(float(v))

        return torch.tensor(y)

    results = {
        mode: {
            f"{s}-{p}": run(s, p, mode)
            for (s, p) in configs
        }
        for mode in modes
    }

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # ---- realizations (left column)
    for i, mode in enumerate(modes):
        ax = axs[i, 0]
        for name, y in results[mode].items():
            ax.plot(y, label=name)

        ax.set_title(f"{mode.upper()} — realizations")
        ax.grid(True)
        if i == 0:
            ax.legend()

    # ---- scaling comparison (right column)
    ref = "df2-zero"

    for i, mode in enumerate(modes):
        ax = axs[i, 1]
        ax.plot(results["core"][ref], label="core", alpha=0.6)
        ax.plot(results["cheap"][ref], label="cheap")
        ax.plot(results["full"][ref], "--", label="full")

        ax.set_title(f"{ref} — scaling comparison")
        ax.grid(True)
        if i == 0:
            ax.legend()

    fig.suptitle("Step Response Comparison (Unified Kernel)")
    plt.tight_layout()
    plt.show()

def compare_transient_diagnostics():
    """
    Transient diagnostics using unified kernel.

    Includes:
        - log-scale error plots
        - linear error plots
        - impulse response comparison
    """

    import matplotlib.pyplot as plt

    T = 150

    step = torch.ones(T)
    impulse = torch.zeros(T)
    impulse[0] = 1.0

    beta = 0.9
    gamma = 0.2
    delta = beta - gamma

    def run(signal, structure, param, mode):
        f = FirstOrderLPF(structure, param, mode)

        q = torch.tensor(0.0)
        y = []

        coeff = gamma if param == "zero" else delta

        for t in range(1, T + 1):
            v, q = f.step(signal[t-1], q, beta, coeff, t)
            y.append(float(v))

        return torch.tensor(y)

    # ---------------------------------------------
    # reference config
    # ---------------------------------------------
    structure = "df2"
    param = "zero"

    y_core  = run(step, structure, param, "core")
    y_cheap = run(step, structure, param, "cheap")
    y_full  = run(step, structure, param, "full")

    # ---------------------------------------------
    # errors
    # ---------------------------------------------
    err_core  = torch.abs(y_core  - y_full)
    err_cheap = torch.abs(y_cheap - y_full)

    # ---------------------------------------------
    # impulse responses (all realizations, full mode)
    # ---------------------------------------------
    configs = [
        ("df2",  "zero"),
        ("df2",  "gap"),
        ("df2t", "zero"),
        ("df2t", "gap"),
    ]

    impulse_responses = {
        f"{s}-{p}": run(impulse, s, p, "full")
        for (s, p) in configs
    }

    # ---------------------------------------------
    # plotting
    # ---------------------------------------------

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # step
    axs[0, 0].plot(y_core, label="core", alpha=0.6)
    axs[0, 0].plot(y_cheap, label="cheap")
    axs[0, 0].plot(y_full, "--", label="full")
    axs[0, 0].set_title("Step Response")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # log error
    axs[0, 1].plot(err_core, label="|core - full|")
    axs[0, 1].plot(err_cheap, label="|cheap - full|")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlim(0, 60)
    axs[0, 1].set_title("Transient Error (log scale)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # linear error
    axs[1, 0].plot(err_core, label="core")
    axs[1, 0].plot(err_cheap, label="cheap")
    axs[1, 0].set_xlim(0, 40)
    axs[1, 0].set_title("Transient Error (linear)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # impulse response
    for name, y in impulse_responses.items():
        axs[1, 1].plot(y, label=name)

    axs[1, 1].set_title("Impulse Response (all realizations)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.suptitle("Transient Diagnostics (Unified Kernel)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_frequency_response()
    compare_step_responses()
    compare_transient_diagnostics()

