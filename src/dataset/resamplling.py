import torch
import torch.nn.functional as F

def _ensure_float(x):
    return x if x.dtype.is_floating_point else x.float()

def _to_nct(x):
    # (N, T, F) -> (N, F, T) for torch.interpolate
    return x.permute(0, 2, 1)

def _to_ntf(x):
    # (N, F, T) -> (N, T, F)
    return x.permute(0, 2, 1)

def _previous_hold_resample(x, tgt_T):
    """
    Step-wise (zero-order hold) down/up-sampling along time using index gather.
    x: (N, T, F)
    """
    N, T, Fdim = x.shape
    # Map target time indices (0..tgt_T-1) to source indices (0..T-1)
    # using floor of the proportional position.
    # When upsampling, this repeats samples; when downsampling, it picks representatives.
    pos = torch.linspace(0, 1, steps=tgt_T, device=x.device)
    src_idx = torch.clamp((pos * (T - 1)).floor().long(), 0, T - 1)  # (tgt_T,)
    # Gather along time dimension
    x_g = x.index_select(dim=1, index=src_idx)  # (N, tgt_T, F)
    return x_g

def _cubic_spline_resample_cpu(x, tgt_T):
    """
    Natural cubic spline via SciPy (CPU), per feature, per node.
    Falls back to linear if SciPy is unavailable.
    x: (N, T, F)
    """
    try:
        import numpy as np
        from scipy.interpolate import CubicSpline
    except Exception:
        # Fallback to linear torch interpolation if SciPy is not present
        return _to_ntf(
            F.interpolate(_to_nct(_ensure_float(x)), size=tgt_T, mode="linear", align_corners=True)
        )

    N, T, Fdim = x.shape
    t_src = np.linspace(0.0, 1.0, T, dtype=np.float64)
    t_tgt = np.linspace(0.0, 1.0, tgt_T, dtype=np.float64)

    # Move to CPU numpy
    x_np = x.detach().cpu().numpy().astype(np.float64)
    out = np.empty((N, tgt_T, Fdim), dtype=np.float64)

    for i in range(N):
        for f in range(Fdim):
            cs = CubicSpline(t_src, x_np[i, :, f], bc_type="natural")
            out[i, :, f] = cs(t_tgt)

    out_t = torch.from_numpy(out).to(x.device, dtype=torch.float32 if not x.dtype.is_floating_point else x.dtype)
    return out_t

def resample_to_common_time(A: torch.Tensor,
                            B: torch.Tensor,
                            target: str = "A",
                            method: str = "linear",):
    """
    Resample tensors A and B along the time axis to share the same number of timesteps.

    Parameters
    ----------
    A : torch.Tensor of shape (N_A, T_A, F_A)
    B : torch.Tensor of shape (N_B, T_B, F_B)
    target : {"A","B"}
        Choose whose timeline length to adopt:
        - "A": both will have T_A timesteps
        - "B": both will have T_B timesteps
    method : {"linear","nearest","previous","spline"}
        Interpolation method along time:
        - "linear": torch 1D linear interpolation (uniform grid)
        - "nearest": nearest-neighbor (uniform grid)
        - "previous": step-wise hold (zero-order hold)
        - "spline": natural cubic spline (requires SciPy; falls back to linear if unavailable)

    Returns
    -------
    A_res, B_res : torch.Tensor
        Resampled tensors with shapes (N_A, T_target, F_A) and (N_B, T_target, F_B)
    """
    assert A.ndim == 3 and B.ndim == 3, "Expected A and B with shape (N, T, F)"
    T_A = A.shape[1]
    T_B = B.shape[1]
    if target.upper() == "A":
        T_target = T_A
    elif target.upper() == "B":
        T_target = T_B
    else:
        raise ValueError("target must be 'A' or 'B'")

    method = method.lower()
    supported = {"linear", "nearest", "previous", "spline"}
    if method not in supported:
        raise ValueError(f"Unsupported method '{method}'. Choose from {supported}.")

    # Identity if already matching
    def _maybe_identity(X, T_src):
        return X if T_src == T_target else None

    # Choose resampler
    def _resample(X):
        T_src = X.shape[1]
        same = _maybe_identity(X, T_src)
        if same is not None:
            return same

        if method == "linear":
            Xnct = _to_nct(_ensure_float(X))
            Y = F.interpolate(Xnct, size=T_target, mode="linear", align_corners=True)
            return _to_ntf(Y).to(X.dtype)
        elif method == "nearest":
            Xnct = _to_nct(_ensure_float(X))
            Y = F.interpolate(Xnct, size=T_target, mode="nearest")
            return _to_ntf(Y).to(X.dtype)
        elif method == "previous":
            return _previous_hold_resample(X, T_target)
        elif method == "spline":
            return _cubic_spline_resample_cpu(X, T_target)
        else:
            raise RuntimeError("Unreachable")

    A_res = _resample(A)
    B_res = _resample(B)
    return A_res, B_res

# ---------- example usage ----------
# A: (913, 1378, 3), B: (121, 1339, 4)
# A_res, B_res = resample_to_common_time(A, B, target="A", method="linear")
# Now: A_res.shape == (913, 1378, 3), B_res.shape == (121, 1378, 4)
