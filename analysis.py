"""Learned ranking of HTN recovery options (LightGBM) vs fixed library order, on a seeded grid-world simulation."""
import json, os, sys
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score

N, N_EP, N_OPT, BUDGET, SEEDS = 15, 4000, 12, 250, (42, 43, 44)  # grid size, episodes, options, expansion budget
OFFS = [(4, 0), (0, 4), (-4, 0), (0, -4), (4, 4), (-4, 4), (4, -4), (-4, -4), (7, 0), (0, 7), (0, -7)]
KEY_SLOT = 5  # fixed library slot of the "fetch key" method (the other 11 are offset waypoints)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def bfs(gl, src, dst, key, limit, kc):
    """Breadth-first search over (cell, has_key); doors (2) need the key. Returns (dist|None, key_at_end, expansions)."""
    k0 = key or src == kc
    seen = {(src[0], src[1], k0)}
    q, n = deque([(src[0], src[1], k0, 0)]), 0
    while q:
        x, y, k, d = q.popleft()
        if (x, y) == dst:
            return d, k, n
        n += 1
        if d >= limit:
            continue
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < N and 0 <= ny < N):
                continue
            c = gl[ny][nx]
            if c == 1 or (c == 2 and not k):
                continue
            s = (nx, ny, k or (nx, ny) == kc)
            if s not in seen:
                seen.add(s)
                q.append((*s, d + 1))
    return None, False, n


def line(a, b):
    return range(a, b + 1) if b >= a else range(a, b - 1, -1)


def make_episode(rng):
    """Original plan = L-shaped path; a wall of obstacles or locked doors then blocks it. Returns state dict."""
    while True:
        g = (rng.random((N, N)) < 0.12).astype(int)
        s, goal = (tuple(int(v) for v in rng.integers(0, N, 2)) for _ in range(2))
        if abs(s[0] - goal[0]) + abs(s[1] - goal[1]) < 8:
            continue
        path = [(x, s[1]) for x in line(s[0], goal[0])] + [(goal[0], y) for y in line(s[1], goal[1])][1:]
        for x, y in path:
            g[y, x] = 0
        i = int(rng.integers(1, len(path) - 1))
        cur, fail = path[i - 1], path[i]
        d = (fail[0] - cur[0], fail[1] - cur[1])
        axis = (abs(d[1]), abs(d[0]))
        up, down = (int(v) for v in rng.integers(0, 5, 2))
        door = bool(rng.random() < 0.5)
        wall = 0
        for t in range(-down, up + 1):
            x, y = fail[0] + t * axis[0], fail[1] + t * axis[1]
            if 0 <= x < N and 0 <= y < N and (x, y) != goal:
                g[y, x] = 2 if door else 1
                wall += 1
        kc = (-1, -1)
        if door:
            free = [(int(x), int(y)) for y, x in np.argwhere(g == 0) if (x, y) not in (cur, goal)]
            kc = free[int(rng.integers(len(free)))]
        gl = g.tolist()
        dist, _, full_cost = bfs(gl, cur, goal, False, 10**9, kc)
        if dist is None:
            continue  # unsolvable: resample
        limit = int(dist * 1.25) + 1  # plan-length bound of the repaired plan
        return dict(g=g, gl=gl, cur=cur, goal=goal, kc=kc, door=door, limit=limit, fail=fail, d=d,
                    axis=axis, up=up, down=down, wall=wall, full_cost=full_cost)


def waypoints(e):
    cur = e["cur"]
    ws = [(min(max(cur[0] + dx, 0), N - 1), min(max(cur[1] + dy, 0), N - 1)) for dx, dy in OFFS]
    ws.insert(KEY_SLOT, e["kc"] if e["door"] else cur)  # no key -> option trivially invalid
    return ws


def run_option(e, w):
    """Symbolic check of one recovery method: cur -> waypoint -> goal within the length bound."""
    gl, cur = e["gl"], e["cur"]
    if w == cur or gl[w[1]][w[0]] != 0:
        return 0, 1
    d1, k1, n1 = bfs(gl, cur, w, False, e["limit"], e["kc"])
    if d1 is None:
        return 0, n1 + 1
    d2, _, n2 = bfs(gl, w, e["goal"], k1, e["limit"] - d1, e["kc"])
    return int(d2 is not None), n1 + n2 + 1


def features(e):
    """Cheap state / failure / option features (no planner calls)."""
    g, cur, goal, kc = e["g"], e["cur"], e["goal"], e["kc"]
    man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    box = lambda a, b: int((g[min(a[1], b[1]):max(a[1], b[1]) + 1, min(a[0], b[0]):max(a[0], b[0]) + 1] != 0).sum())
    dcg = man(cur, goal)
    base = [int(e["door"]), e["wall"], e["up"], e["down"], *cur, goal[0] - cur[0], goal[1] - cur[1], *e["d"], dcg,
            e["limit"] - dcg, float((g == 1).mean()), kc[0] - cur[0] if e["door"] else 0, kc[1] - cur[1] if e["door"] else 0]
    rows = []
    for o, w in enumerate(waypoints(e)):
        d1, d2 = man(cur, w), man(w, goal)
        rel = (w[0] - e["fail"][0], w[1] - e["fail"][1])
        rows.append(base + [o, int(o == KEY_SLOT), w[0] - cur[0], w[1] - cur[1], int(g[w[1], w[0]] != 0), d1, d2,
                            d1 + d2 - dcg, e["limit"] - d1 - d2, rel[0] * e["d"][0] + rel[1] * e["d"][1],
                            abs(rel[0] * e["axis"][0] + rel[1] * e["axis"][1]), box(cur, w), box(w, goal)])
    return rows


def evaluate(order, valid, cost, full):
    """Try options in `order` with the same planner costs; fall back to full replan if none works."""
    n = len(valid)
    v, c = np.take_along_axis(valid, order, 1), np.take_along_axis(cost, order, 1)
    cum = c.cumsum(1)
    tot = np.where(v.any(1), cum[np.arange(n), v.argmax(1)], cum[:, -1] + full)
    return dict(cost=float(tot.mean()), succ=float((tot <= BUDGET).mean()), first=float(v[:, 0].mean()))


def main():
    rng = np.random.default_rng(42)
    print("Simulating episodes...", flush=True)
    X, V, C, F = [], [], [], []
    for _ in range(N_EP):
        e = make_episode(rng)
        res = [run_option(e, w) for w in waypoints(e)]
        X.append(features(e)); V.append([r[0] for r in res]); C.append([r[1] for r in res]); F.append(e["full_cost"])
    X, V, C, F = np.array(X, float), np.array(V), np.array(C), np.array(F)
    nf = X.shape[2]
    tr, te = train_test_split(np.arange(N_EP), test_size=0.3, random_state=42)  # single split by episode
    print(f"train={len(tr)} test={len(te)} option success rate (train)={V[tr].mean():.3f}", flush=True)
    if V[tr].min() == V[tr].max() or V[te].sum() == 0:
        sys.exit("degenerate simulation: no usable labels")

    # Baseline: fixed library order, then full replan
    base = evaluate(np.tile(np.arange(N_OPT), (len(te), 1)), V[te], C[te], F[te])

    # Improved: class-balanced LightGBM, hyper-parameters tuned by grouped CV on the training episodes only
    Xtr, ytr, grp = X[tr].reshape(-1, nf), V[tr].ravel(), np.repeat(tr, N_OPT)
    mk = lambda seed: lgb.LGBMClassifier(class_weight="balanced", n_estimators=150, random_state=seed, verbose=-1, n_jobs=2)
    gs = GridSearchCV(mk(42), {"num_leaves": [7, 31], "learning_rate": [0.05, 0.1]}, scoring="average_precision",
                      cv=GroupKFold(3), n_jobs=1)
    gs.fit(Xtr, ytr, groups=grp)
    print("best params:", gs.best_params_, flush=True)
    runs, aucs = [], []
    for sd in SEEDS:
        m = mk(sd).set_params(**gs.best_params_).fit(Xtr, ytr)
        P = m.predict_proba(X[te].reshape(-1, nf))[:, 1].reshape(-1, N_OPT)
        runs.append(evaluate(np.argsort(-P, axis=1, kind="stable"), V[te], C[te], F[te]))
        aucs.append(roc_auc_score(V[te].ravel(), P.ravel()))
        print(f"seed {sd}: mean expansions={runs[-1]['cost']:.2f}", flush=True)
    mean = lambda k: float(np.mean([r[k] for r in runs]))
    std = lambda k: float(np.std([r[k] for r in runs]))

    metrics = {
        "baseline_score": base["cost"], "improved_score": mean("cost"), "improvement": base["cost"] - mean("cost"),
        "baseline_score_std": 0.0, "improved_score_std": std("cost"),
        "baseline_success_rate": base["succ"], "improved_success_rate": mean("succ"), "improved_success_rate_std": std("succ"),
        "baseline_first_option_works": base["first"], "improved_first_option_works": mean("first"),
        "improved_first_option_works_std": std("first"),
        "improved_option_auc": float(np.mean(aucs)), "share_episodes_with_valid_option_test": float(V[te].any(1).mean()),
        "expansion_budget": float(BUDGET),
    }
    if not all(np.isfinite(list(metrics.values()))):
        sys.exit("non-finite metrics")
    # improved first-option share needs the ranked first option; recompute is already in evaluate via the ranked order
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].bar(["fixed order", "LightGBM rank"], [metrics["baseline_score"], metrics["improved_score"]], color=["gray", "tab:blue"])
    ax[0].set_title("Mean node expansions (lower better)")
    ax[1].bar(["fixed order", "LightGBM rank"], [metrics["baseline_success_rate"], metrics["improved_success_rate"]],
              color=["gray", "tab:blue"])
    ax[1].set_title("Success within budget")
    fig.tight_layout(); fig.savefig("figures/comparison.png", dpi=120)

    out = {
        "question": "Can a LightGBM model that ranks recovery options reduce the symbolic search effort needed to repair a broken hierarchical plan, compared with trying options in a fixed library order?",
        "method": "Simulated grid-world HTN plan failures; 12 candidate recovery methods checked by a BFS planner; LightGBM predicts which options succeed and options are tried in ranked order.",
        "dataset": {"name": "Simulated grid-world HTN plan-failure episodes (seeded)", "source_url": "generated in analysis.py (no download)",
                    "n_rows": N_EP, "n_train": len(tr), "n_test": len(te), "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "mean node expansions until a valid recovery is found (lower is better)",
        "higher_is_better": False,
        "baseline": "Standard symbolic recovery stand-in: try the 12 recovery methods in fixed library order, then fall back to a full replan.",
        "improved": "Class-balanced, grouped-CV-tuned LightGBM predicts option success from state/failure/option features; options are tried in ranked order with the same planner, costs and full-replan fallback.",
        "metrics": metrics,
        "notes": "Synthetic data: results depend on the simulator design (grid size, wall shapes, length bound 1.25x optimal+1, arbitrary fixed library order) and say nothing about the paper's system. Option costs are precomputed once per episode and shared by both methods. Baseline is deterministic (std 0); LightGBM std is over seeds 42-43-44 with hyper-parameters tuned once (seed 42) on training episodes. Budget is a fixed constant, not tuned.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
