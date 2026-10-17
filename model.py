"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - short_conv
def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    # TODO: accumulate each kernel tap with a shifted slice add
    T, d = x.shape
    K = w.shape[0]
    y = np.zeros_like(x)
    for j in range(K):
        shift = K - 1 - j
        if shift == 0:
            y += w[j] * x
        else:
            y[shift:] += w[j] * x[:-shift]
    return y

# Step 2 - kda_qkv
# ── Step 002 kda_qkv   ──
def swish(x):
    return x / (1 + np.exp(-x))

def l2_norm(x):
    return x / np.sqrt(np.sum(x**2, axis=-1, keepdims=True) + 1e-6)

def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    
    Wq, Wk, Wv = params['Wq'], params['Wk'], params['Wv']
    cq, ck, cv = params['cq'], params['ck'], params['cv']
    q = short_conv(x @ Wq, cq)
    q = l2_norm(swish(q))
    k = short_conv(x @ Wk, ck)
    k = l2_norm(swish(k))
    v = short_conv(x @ Wv, cv)
    v = swish(v)
    return q, k, v

# Step 3 - kda_gates
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    beta = sigmoid(x @ params['wb'] + params['bb'])
    z = x @ params['Wd1'] @ params['Wd2'] + params['ba']
    return beta, z

# Step 4 - lower_bounded_decay
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    return np.exp(g_min * sigmoid(np.exp(A) * z))

# Step 5 - kda_state_update
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    return (np.eye(S.shape[0]) - beta * k[:, np.newaxis] * k[np.newaxis, :]) @ np.diag(alpha) @ S + beta * k[:, np.newaxis] * v[np.newaxis, :]

# Step 6 - kda_recurrence
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T = q.shape[0]
    O = np.zeros((T, v.shape[-1]))
    S = S0 if S0 is not None else np.zeros((k.shape[-1], v.shape[-1]))
    for t in range(T):
        S = kda_state_update(S, k[t], v[t], alpha[t], beta[t])
        O[t] = S.T @ q[t]
    return O, S

# Step 7 - cumulative_decay
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    return np.cumprod(alpha, axis=0)

# Step 8 - chunk_pseudo_values
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    C = k.shape[0]
    gamma = cumulative_decay(alpha) # (C, dk)
    Khat = k * gamma # (C, dk)
    Kcheck = k / gamma # (C, dk)
    beta_diag = np.diag(beta) # (C, C)
    U = np.linalg.solve(np.eye(C) + beta_diag @ np.tril(Khat @ Kcheck.T, -1), beta_diag @ (v - Khat @ S0)) # no idea what this is going to do
    return U

# Step 9 - kda_chunkwise
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. 
    Returns (O, S_final).
    """
    O = np.zeros((q.shape[0], v.shape[-1]))
    S = S0 if S0 is not None else np.zeros((k.shape[-1], v.shape[-1]))
    for i in range(0, len(q), chunk_size):
        q_chunk = q[i:i+chunk_size]
        k_chunk = k[i:i+chunk_size]
        v_chunk = v[i:i+chunk_size]
        alpha_chunk = alpha[i:i+chunk_size]
        beta_chunk = beta[i:i+chunk_size]

        gamma = cumulative_decay(alpha_chunk) # (C, dk)
        q_hat = q_chunk * gamma
        k_check = k_chunk / gamma # (C, dk)

        U_chunk = chunk_pseudo_values(k_chunk, v_chunk, alpha_chunk, beta_chunk, S)
        O_chunk = q_hat @ S + np.tril(q_hat @ k_check.T) @ U_chunk
        S = gamma[-1][:, np.newaxis] * (S + k_check.T @ U_chunk)

        O[i:i+chunk_size] = O_chunk
    return O, S

# Step 10 - kda_output_gate
def rms_norm(x):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)

def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    g = sigmoid(x @ Wg)
    return (g * rms_norm(o)) @ Wo

# Step 11 - mla_compress_reconstruct
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    T = x.shape[0]
    c = x @ Wc # (T, r)
    K = (c @ Wk_up).reshape(T, n_heads, -1) # (T, n_heads, dh)
    V = (c @ Wv_up).reshape(T, n_heads, -1) # (T, n_heads, dh)
    return c, K, V

# Step 12 - nope_attention
def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)
    
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T = x.shape[0]
    c, K, V = mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads)
    Q = (x @ Wq).reshape(T, n_heads, -1) # (T, n_heads, dh)

    dh = Q.shape[-1]
    scores = np.einsum('thd,jhd->thj', Q, K) / np.sqrt(dh)
    mask = np.triu(np.ones((T, T)), k=1)
    scores = scores - 1e10 * mask[:, None, :]
    attn = softmax(scores)
    out = np.einsum('thj,jhd->thd', attn, V) # (T, n_heads, dh)
    return out.reshape(T, -1)

# Step 13 - mla_output_gate
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    g = sigmoid(x @ Wg)
    return (g * o) @ Wo

# Step 14 - hybrid_schedule
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    return ['KDA', 'KDA', 'KDA', 'MLA'] * n_repeats + ['MLA']

# Step 15 - attnres_weights
def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    logits = np.stack([rms_norm(s) @ pseudo_q for s in sources], axis=0)  # (n, T)
    return softmax(logits, axis=0)

# Step 16 - attnres_full
def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    w = attnres_weights(pseudo_q, sources)
    return np.sum(w[:,:,None] * np.stack(sources, axis=0), axis=0)

# Step 17 - block_partial_sums
def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    partial_sums = []
    accum = np.zeros_like(layer_outputs[0])
    for output in layer_outputs:
        accum += output
        partial_sums.append(accum.copy())
    return partial_sums

# Step 18 - attnres_block
def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).
    - b_n are the sum of the blocks layer outputs, b_n = o_n1 + o_n2 + ... + o_n4
    - partial is the current block's accumulated outputs partial_n3 = o_n1 + o_n2

    partial is None for the first layer of a block. Returns (T, d).
    """
    sources = block_reps if partial is None else block_reps + [partial]
    return attnres_full(pseudo_q, sources)

# Step 19 - situ_glu
def sigmoid_safe(x):
    return 0.5 * (1 + np.tanh(0.5 * x))

def softcap(x, b):
    return b * np.tanh(x / b)

def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """

    return (softcap(x @ Wg, beta1) * sigmoid_safe(x @ Wg)) * softcap(x @ Wu, beta2)

# Step 20 - route_topk
def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    s = sigmoid(x @ Wr)  # (T, dr)
    idx = np.argsort(-(s + bias), axis=-1, kind="stable")[:, :k]  # (T, k)
    selected = np.take_along_axis(s, idx, axis=-1)  # (T, k) raw scores at top-k
    p = selected / np.sum(selected, axis=-1, keepdims=True)
    return s, idx, p

# Step 21 - routed_experts
def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    idx: (T, k)
    p: (T, k)
    experts: list of (Wg, Wu)

    return aggregated routed representation u: (T, l)
    """
    T = z.shape[0]
    k = p.shape[1]
    u = np.zeros_like(z) # (T, l)
    for t in range(T):
        for j in range(k):
            # (1, l)
            u[t] += p[t, j] * situ_glu(z[t], *experts[idx[t, j]])

    return u

# Step 22 - stable_latent_moe
def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    s, idx, p = route_topk(x, params['Wr'], params['bias'], params['k'])
    z = x @ params['Wdown']
    u = routed_experts(z, idx, p, params['experts'])
    shared = sum(situ_glu(x, Wg, Wu) for Wg, Wu in params['shared'])
    return shared + rms_norm(u) @ params['Wup']

# Step 23 - topk_cutoffs
def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    
    order = np.argsort(-(s + bias), axis=-1, kind="stable")
    routes = order[:, :k]

    # gather biased scores
    cutoffs = (s + bias)[np.arange(s.shape[0]), order[:, k]]
    return routes, cutoffs

# Step 24 - quantile_balance_update
def quantile_balance_update(s, bias, k):
    """QB (Eq. 14): bhat_j = -(the (q+1)-th largest of s[:, j] - cutoffs),
    q = m*k // n; return bhat - mean(bhat).
    """
    m, n = s.shape
    _, cutoffs = topk_cutoffs(s, bias, k)
    margins = s - cutoffs[:, None]
    q = m * k // n
    bhat = np.sort(-margins, axis=0)[q]
    return bhat - np.mean(bhat)

# Step 25 - histogram_quantile
def histogram_quantile(x, n_bins, lo, hi, q_frac):
    """Quantile from pooled bin counts; error <= (hi - lo) / n_bins.

    Return the right edge of the first bin whose cumulative count reaches
    q_frac * len(x).
    """
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    b = np.searchsorted(np.cumsum(counts), q_frac * len(x))
    return edges[b + 1]

# Step 26 - newton_schulz
def newton_schulz(G, n_iters=5):
    """Muon's Newton-Schulz orthogonalization (a,b,c = 3.4445, -4.7750, 2.0315).

    Normalize by the Frobenius norm (+1e-7), iterate the quintic, transpose
    handling for tall matrices. Singular values -> 1.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G, 'fro') + 1e-7)
    for _ in range(n_iters):
        A= X @ X.T
        B = b*A + c*A @ A 
        X = a*X + B @ X
    return X

# Step 27 - per_head_muon
def per_head_muon(M, n_heads, n_iters=5):
    """Split M (d, H*dh) into H column blocks, newton_schulz each, re-concatenate."""
    M_heads = [newton_schulz(M_head, n_iters) for M_head in np.split(M, n_heads, axis=1)]
    return np.concatenate(M_heads, axis=-1)

# Step 28 - mini_k3_forward
def mini_k3_forward(tokens, seed, block_size=1):
    """Miniature Kimi K3 forward pass. Returns (T, 17) logits.

    block_size counts hybrid layers (attention + MoE) per residual block.
    The embedding is the first source. Each layer attends over completed blocks
    and the current partial, then contributes attention_out + MoE(attention_out).
    Completed blocks store sum(layer_outputs); partial contains only the current block.
    The default keeps one representation per layer; w_final is the final AttnRes query.

    Parameter creation order (all rng.normal(0.0, 0.3, size=...)):
      emb (17, 16)
      then for each layer of hybrid_schedule(1):
        wq_res (16,)
        KDA: Wq (16,8), Wk (16,8), Wv (16,8), cq (3,8), ck (3,8), cv (3,8),
             wb (16,), Wd1 (16,4), Wd2 (4,8), ba (8,), Wg (16,8), Wo (8,16)
             (bb = 0.0 and A = 0.0 are constants, not drawn)
        MLA: Wq (16,16), Wc (16,8), Wk_up (8,16), Wv_up (8,16),
             Wg (16,16), Wo (16,16)
        MoE: Wdown (16,8), Wup (8,16), Wr (16,8), bias = zeros(8), k=2,
             experts: 8 x (Wg (8,8), Wu (8,8)) drawn gate-then-up per expert,
             shared: 2 x (Wg (16,16), Wu (16,16)) drawn gate-then-up
      w_final (16,)
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(seed)
    def W(*shape):
        return rng.normal(0.0, 0.3, size=shape)

    emb = W(17, 16)
    x = emb[tokens]

    block_reps = [x]
    layer_outputs = []
    partial = None

    schedule = hybrid_schedule(1)
    for layer_idx, layer in enumerate(schedule):
        wq_res = W(16)
        x_in = attnres_block(wq_res, block_reps, partial)
        if layer == 'KDA':
            kda = {
                'Wq': W(16, 8), 'Wk': W(16, 8), 'Wv': W(16, 8),
                'cq': W(3, 8), 'ck': W(3, 8), 'cv': W(3, 8),
                'wb': W(16), 'Wd1': W(16, 4), 'Wd2': W(4, 8), 'ba': W(8),
                'Wg': W(16, 8), 'Wo': W(8, 16), 'bb': 0.0,
            }
            q, k, v = kda_qkv(x_in, kda)
            beta, z = kda_gates(x_in, kda)
            alpha = lower_bounded_decay(z, 0.0)
            O, _ = kda_recurrence(q, k, v, alpha, beta)
            o = kda_output_gate(O, x_in, kda['Wg'], kda['Wo'])
        else:
            mla = {
                'Wq': W(16, 16), 'Wc': W(16, 8),
                'Wk_up': W(8, 16), 'Wv_up': W(8, 16),
                'Wg': W(16, 16), 'Wo': W(16, 16),
            }
            O = nope_attention(
                x_in, mla['Wq'], mla['Wc'], mla['Wk_up'], mla['Wv_up'], n_heads=8,
            )
            o = mla_output_gate(O, x_in, mla['Wg'], mla['Wo'])

        moe = {
            'Wdown': W(16, 8), 'Wup': W(8, 16), 'Wr': W(16, 8),
            'bias': np.zeros(8), 'k': 2,
            'experts': [(W(8, 8), W(8, 8)) for _ in range(8)],
            'shared': [(W(16, 16), W(16, 16)), (W(16, 16), W(16, 16))],
        }
        o = o + stable_latent_moe(o, moe)
        layer_outputs.append(o)
        partial = sum(layer_outputs)
        if (layer_idx + 1) % block_size == 0 or layer_idx + 1 == len(schedule):
            block_reps.append(partial)
            layer_outputs = []
            partial = None

    w_final = W(16)
    x = attnres_block(w_final, block_reps, partial)
    return x @ emb.T

