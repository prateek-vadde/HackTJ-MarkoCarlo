from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import yfinance as yf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    date: str


@app.get("/")
async def root():
    return {"status": "ok", "usage": "POST /analyze with {\"date\": \"YYYY-MM-DD\"}"}


def run_phase1(req_date: str):
    """Run the full Phase 1 geometric signal extraction pipeline.
    Returns a dict with all computed values for reuse by Phase 2.
    """
    # Step 1 — Data Acquisition
    try:
        end_date = datetime.strptime(req_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    start_date = end_date - timedelta(days=21)
    df = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if df.empty or len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough datapoints (need >= 10).")

    prices = df["Close"].values.astype(np.float64).flatten()
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    # Step 2 — Wavefunction Construction
    z = hilbert(prices)
    psi = z / np.linalg.norm(z)
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)

    # Step 3 — Fubini-Study Curvature (scalar part of QGT)
    dpsi = np.gradient(psi)
    n = len(psi)
    curvature = np.zeros(n)
    for i in range(n):
        dpsi_dpsi = np.abs(dpsi[i]) ** 2
        psi_dpsi = np.conj(psi[i]) * dpsi[i]
        curvature[i] = abs(dpsi_dpsi - abs(psi_dpsi) ** 2)

    # Step 4 — Crisis probability (volatility regime detection)
    # Spearman ρ = 0.723 with forward volatility. DO NOT MODIFY.
    curvature_smooth = gaussian_filter1d(curvature, sigma=1.5)
    curv_normalized = curvature_smooth / (np.sum(curvature_smooth) + 1e-10)
    entropy = -np.sum(curv_normalized * np.log(curv_normalized + 1e-10))
    max_entropy = np.log(n)
    concentration = 1.0 - (entropy / max_entropy)

    risk_score = float(np.mean(curvature))
    risk_normalized = 1.0 / (1.0 + np.exp(-2.0 * (np.log10(risk_score + 1e-10) + 4.5)))

    predicted_crisis_probability = float(risk_normalized * concentration)

    # Step 5 — Phase dynamics (directionality)
    phase = np.unwrap(np.angle(psi))
    inst_freq = np.gradient(phase)
    freq_acceleration = np.gradient(inst_freq)
    direction = float(np.sign(inst_freq[-1]))
    reversal_risk = float(-freq_acceleration[-1] * direction)

    # Step 6 — Directional crisis (magnitude × direction)
    directional_crisis = float(predicted_crisis_probability * np.sign(-reversal_risk))

    # Step 7 — Berry curvature (imaginary part of QGT)
    berry_curvature = np.imag(
        np.outer(np.conj(dpsi), dpsi) - np.outer(np.conj(psi), psi)
    )

    return {
        "dates": dates,
        "prices": prices,
        "psi": psi,
        "psi_real": psi_real,
        "psi_imag": psi_imag,
        "dpsi": dpsi,
        "n": n,
        "curvature": curvature,
        "risk_score": risk_score,
        "concentration": concentration,
        "predicted_crisis_probability": predicted_crisis_probability,
        "inst_freq": inst_freq,
        "freq_acceleration": freq_acceleration,
        "direction": direction,
        "reversal_risk": reversal_risk,
        "directional_crisis": directional_crisis,
        "berry_curvature": berry_curvature,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    p = run_phase1(req.date)
    return {
        "dates": p["dates"],
        "prices": p["prices"].tolist(),
        "psi_real": p["psi_real"].tolist(),
        "psi_imag": p["psi_imag"].tolist(),
        "curvature": p["curvature"].tolist(),
        "risk_score": p["risk_score"],
        "concentration": p["concentration"],
        "predicted_crisis_probability": p["predicted_crisis_probability"],
        "phase_momentum": float(p["inst_freq"][-1]),
        "inst_freq": p["inst_freq"].tolist(),
        "freq_acceleration": p["freq_acceleration"].tolist(),
        "direction": p["direction"],
        "reversal_risk": p["reversal_risk"],
        "directional_crisis": p["directional_crisis"],
        "berry_curvature": p["berry_curvature"].tolist(),
        "forward_curvature_distribution": p["curvature"].tolist(),
        "forward_curvature_bin_edges": list(range(p["n"] + 1)),
    }


# ---------------------------------------------------------------------------
# Phase 2 — Neural Monte Carlo RL Path Generation
# ---------------------------------------------------------------------------

class NeuralManifoldWalker(nn.Module):
    def __init__(self, n):
        super().__init__()
        # Action dimension: 2n (tangent real+imag) + 3 (drift_scale, vol_scale, step_scale)
        self.action_dim = 2 * n + 3
        # Output: mean (action_dim) + log_std (action_dim)
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * self.action_dim),
        )

    def forward(self, x):
        raw = self.net(x)
        mean = raw[:, :self.action_dim]
        log_std = raw[:, self.action_dim:]
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -3.0, 1.0)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        action = mean + std * noise
        # Per-dimension log probability: sum of log N(action_i | mean_i, std_i)
        log_prob = -0.5 * (((action - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob


def construct_psi_from_prices(price_window):
    if len(price_window) < 8:
        return None
    z = hilbert(np.array(price_window, dtype=np.float64))
    norm = np.linalg.norm(z)
    if norm < 1e-10:
        return None
    return z / norm


def compute_berry_magnitude(psi_window):
    dpsi_w = np.gradient(psi_window)
    return float(np.mean(np.abs(
        np.conj(psi_window) * np.gradient(dpsi_w)
        - dpsi_w * np.gradient(np.conj(psi_window))
    )))


def compute_step_geometry(price_window):
    """Compute geometric state from a price window for sequential path stepping.
    Returns dict with all signals needed to feed back into the walker,
    or None if the window is too small / degenerate.
    """
    if len(price_window) < 8:
        return None
    pw = np.array(price_window, dtype=np.float64)
    z = hilbert(pw)
    norm = np.linalg.norm(z)
    if norm < 1e-10:
        return None
    psi_w = z / norm
    dpsi_w = np.gradient(psi_w)
    n_w = len(psi_w)

    # Curvature
    curvature_w = np.zeros(n_w)
    for i in range(n_w):
        dd = np.abs(dpsi_w[i]) ** 2
        pd = np.conj(psi_w[i]) * dpsi_w[i]
        curvature_w[i] = abs(dd - abs(pd) ** 2)

    # Risk score
    risk_score_w = float(np.mean(curvature_w))

    # Crisis prob (smoothed entropy)
    curvature_smooth = gaussian_filter1d(curvature_w, sigma=1.5)
    curv_norm = curvature_smooth / (np.sum(curvature_smooth) + 1e-10)
    entropy = -np.sum(curv_norm * np.log(curv_norm + 1e-10))
    max_entropy = np.log(n_w)
    concentration = 1.0 - (entropy / max_entropy)
    risk_normalized = 1.0 / (1.0 + np.exp(-2.0 * (np.log10(risk_score_w + 1e-10) + 4.5)))
    crisis_prob_w = float(risk_normalized * concentration)

    # Phase dynamics
    phase = np.unwrap(np.angle(psi_w))
    inst_freq_w = np.gradient(phase)
    freq_accel_w = np.gradient(inst_freq_w)
    direction = float(np.sign(inst_freq_w[-1]))
    reversal_risk_w = float(-freq_accel_w[-1] * direction)

    # Berry magnitude
    berry_mag_w = float(np.mean(np.abs(np.imag(
        np.conj(dpsi_w) * np.gradient(dpsi_w)
        - np.outer(np.conj(psi_w), psi_w).diagonal() * np.gradient(dpsi_w)
    ))))

    # Empirical drift/vol from this window
    log_rets = np.diff(np.log(pw))
    emp_drift = float(np.mean(log_rets))
    emp_vol = float(np.std(log_rets))

    return {
        "psi": psi_w,
        "dpsi": dpsi_w,
        "n": n_w,
        "curvature": curvature_w,
        "risk_score": risk_score_w,
        "crisis_prob": crisis_prob_w,
        "inst_freq": inst_freq_w,
        "freq_acceleration": freq_accel_w,
        "reversal_risk": reversal_risk_w,
        "berry_magnitude": berry_mag_w,
        "empirical_drift": emp_drift,
        "empirical_vol": emp_vol,
    }


@app.post("/analyze-paths")
async def analyze_paths(req: AnalyzeRequest):
    # Step 1 — Run full Phase 1 pipeline, reuse all computed values
    p = run_phase1(req.date)

    dates = p["dates"]
    prices = p["prices"]
    psi = p["psi"]
    dpsi = p["dpsi"]
    n = p["n"]
    curvature = p["curvature"]
    risk_score = p["risk_score"]
    inst_freq = p["inst_freq"]
    reversal_risk = p["reversal_risk"]
    crisis_prob = p["predicted_crisis_probability"]

    # Berry magnitude for current state
    berry_magnitude_current = float(np.mean(np.abs(np.imag(
        np.conj(dpsi) * np.gradient(dpsi)
        - np.outer(np.conj(psi), psi).diagonal() * np.gradient(dpsi)
    ))))

    # Empirical drift and vol from actual price returns (financial units)
    log_returns = np.diff(np.log(prices))
    empirical_drift = float(np.mean(log_returns))       # daily mean log-return
    empirical_vol = float(np.std(log_returns))           # daily volatility
    # Geometric signal modulates direction: inst_freq sign gives trend direction
    drift_direction = float(np.sign(inst_freq[-1]))

    # Step 2 — NeuralManifoldWalker
    walker = NeuralManifoldWalker(n)
    optimizer = torch.optim.Adam(walker.parameters(), lr=0.001)

    # Step 4 — RL Monte Carlo loop
    N_PATHS = 1000
    FORWARD_DAYS = 10
    WARMUP_STEPS = 150

    geo_features = np.array([
        risk_score / (risk_score + 1e-8),
        berry_magnitude_current / (berry_magnitude_current + 1e-8),
        np.tanh(reversal_risk),
        crisis_prob,
    ], dtype=np.float32)
    geo_tensor = torch.tensor(geo_features, dtype=torch.float32).unsqueeze(0)

    # Kinematic forward projection: Δφ = ω·t + ½·α·t²
    freq_accel = p["freq_acceleration"]
    expected_phase_change = inst_freq[-1] * FORWARD_DAYS + 0.5 * freq_accel[-1] * FORWARD_DAYS**2
    phase_direction = float(np.sign(expected_phase_change))

    accepted_paths = []
    running_berry_mean = 0.0
    running_berry_var = 0.0
    running_dir_mean = 0.0
    running_dir_var = 0.0
    running_count = 0

    for step in range(N_PATHS):
        # Sample action from learned distribution
        action, log_prob = walker.sample(geo_tensor)
        action_np = action.detach().numpy().flatten()

        tangent_real = action_np[:n]
        tangent_imag = action_np[n:2*n]
        tangent = tangent_real + 1j * tangent_imag

        # Learned parameter scales via sigmoid/tanh on sampled values
        drift_scale = float(np.tanh(action_np[2*n]))        # [-1, 1]
        vol_scale = float(2.0 / (1.0 + np.exp(-action_np[2*n+1])))  # [0, 2]
        step_scale = float(1.0 / (1.0 + np.exp(-action_np[2*n+2])))  # [0, 1]

        # Project tangent onto manifold: remove component parallel to psi
        tangent = tangent - np.dot(psi.conj(), tangent) * psi
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            continue
        tangent = tangent / tangent_norm

        # Generate forward path by walking manifold step by step
        path_prices = list(prices.copy())
        psi_current_step = psi.copy()
        path_berry_magnitudes = []

        for day in range(FORWARD_DAYS):
            # Step size: base from phase velocity, scaled by learned parameter
            base_step = float(np.abs(inst_freq[-1])) * np.pi / 4
            step_size = np.clip(base_step * step_scale, 0.01, 0.3)

            # Geodesic step on unit sphere
            psi_next = (np.cos(step_size) * psi_current_step
                        + np.sin(step_size) * tangent)
            psi_next = psi_next / np.linalg.norm(psi_next)

            # Recover price: empirical drift/vol in financial units
            # Geometric signal modulates via learned scales
            # psi_next real part gives manifold-consistent oscillation direction
            price_oscillation = np.real(psi_next)
            price_oscillation = price_oscillation - np.mean(price_oscillation)  # zero-center
            price_oscillation = price_oscillation / (np.max(np.abs(price_oscillation)) + 1e-10)
            manifold_signal = float(price_oscillation[-1])

            # Daily return = learned_drift * empirical_drift + learned_vol * empirical_vol * (manifold + noise)
            drift = drift_scale * empirical_drift
            vol = vol_scale * empirical_vol
            noise = np.random.randn()  # per-step stochasticity
            daily_return = drift + vol * (manifold_signal + noise)
            new_price = path_prices[-1] * np.exp(daily_return)
            path_prices.append(new_price)

            # Compute Berry magnitude of this step's wavefunction
            window = path_prices[-(len(prices)):]
            psi_window = construct_psi_from_prices(window)
            if psi_window is not None:
                bm = compute_berry_magnitude(psi_window)
                path_berry_magnitudes.append(bm)
                psi_current_step = psi_window
            else:
                path_berry_magnitudes.append(berry_magnitude_current)

        # REWARD: Berry magnitude (volatility) + directional consistency
        path_return = (path_prices[-1] - path_prices[len(prices)]) / path_prices[len(prices)]
        if len(path_berry_magnitudes) > 0:
            berry_along_path = np.array(path_berry_magnitudes)
            delta_berry = berry_along_path[-1] - berry_magnitude_current
            berry_reward = float(delta_berry / (berry_magnitude_current + 1e-8))
        else:
            berry_reward = 0.0
        # Direction reward: paths aligned with predicted geometric flow
        direction_reward = phase_direction * path_return / (empirical_vol * np.sqrt(FORWARD_DAYS) + 1e-8)

        # Update running stats (Welford's online algorithm)
        running_count += 1
        delta_b = berry_reward - running_berry_mean
        running_berry_mean += delta_b / running_count
        running_berry_var += delta_b * (berry_reward - running_berry_mean)
        delta_d = direction_reward - running_dir_mean
        running_dir_mean += delta_d / running_count
        running_dir_var += delta_d * (direction_reward - running_dir_mean)

        # Combine as z-scores: unit-free, equal weight by construction
        berry_std = np.sqrt(running_berry_var / running_count) if running_count > 1 else 1.0
        dir_std = np.sqrt(running_dir_var / running_count) if running_count > 1 else 1.0
        berry_z = (berry_reward - running_berry_mean) / (berry_std + 1e-8)
        direction_z = (direction_reward - running_dir_mean) / (dir_std + 1e-8)
        # Weight direction by geometry's own directional confidence
        direction_confidence = float(np.tanh(abs(expected_phase_change)))
        reward = berry_z + direction_confidence * direction_z

        # MH acceptance: path must be geometrically consistent with current psi
        if psi_window is not None:
            overlap = min(np.abs(np.dot(psi.conj(), psi_current_step)), 1.0)
            fs_distance = float(np.arccos(overlap))
            mh_acceptance = 1.0 if fs_distance < np.pi / 6 else 0.0
        else:
            mh_acceptance = 0.0

        if np.random.rand() < mh_acceptance:
            accepted_paths.append(path_prices[len(prices):])

        # REINFORCE update after warmup
        if step >= WARMUP_STEPS:
            # reward is already zero-mean (z-scored), so advantage = reward
            advantage = reward

            # log_prob from sample() has per-dimension gradients
            loss = -log_prob.squeeze() * advantage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(walker.parameters(), max_norm=1.0)
            optimizer.step()

    # Fallback if fewer than 10 paths accepted
    if len(accepted_paths) < 10:
        for _ in range(50):
            path = [float(prices[-1])]
            for _ in range(FORWARD_DAYS):
                path.append(path[-1] * np.exp(empirical_drift + empirical_vol * np.random.randn()))
            accepted_paths.append(path[1:])

    # Step 5 — Compute path statistics
    accepted_paths = np.array(accepted_paths)
    initial_price = float(prices[-1])

    forward_returns = (accepted_paths[:, -1] - initial_price) / initial_price

    drawdowns = np.array([
        (np.min(path) - path[0]) / (path[0] + 1e-8)
        for path in accepted_paths
    ])

    path_divergence = float(np.std(accepted_paths[:, -1]) / (initial_price + 1e-8))

    predicted_mean = np.mean(accepted_paths, axis=0).tolist()
    predicted_upper = np.percentile(accepted_paths, 75, axis=0).tolist()
    predicted_lower = np.percentile(accepted_paths, 25, axis=0).tolist()
    expected_return = float(np.mean(forward_returns))
    return_std = float(np.std(forward_returns))
    expected_drawdown = float(np.mean(drawdowns))
    worst_case_drawdown = float(np.percentile(drawdowns, 5))
    crisis_path_fraction = float(np.mean(drawdowns < -0.05))

    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    forward_dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(FORWARD_DAYS + 1)
    ]

    acceptance_rate = float(len(accepted_paths) / N_PATHS)

    # Step 6 — Return
    return {
        # Phase 1 fields
        "dates": dates,
        "prices": prices.tolist(),
        "psi_real": p["psi_real"].tolist(),
        "psi_imag": p["psi_imag"].tolist(),
        "curvature": curvature.tolist(),
        "risk_score": risk_score,
        "crisis_prob": crisis_prob,
        "berry_magnitude": berry_magnitude_current,
        "reversal_risk": reversal_risk,
        # Phase 2 MC path fields
        "forward_dates": forward_dates,
        "mc_paths": accepted_paths[:20].tolist(),
        "predicted_mean": predicted_mean,
        "predicted_upper": predicted_upper,
        "predicted_lower": predicted_lower,
        "expected_return": expected_return,
        "return_std": return_std,
        "expected_drawdown": expected_drawdown,
        "worst_case_drawdown": worst_case_drawdown,
        "crisis_path_fraction": crisis_path_fraction,
        "path_divergence": path_divergence,
        "n_accepted_paths": len(accepted_paths),
        "acceptance_rate": acceptance_rate,
    }
