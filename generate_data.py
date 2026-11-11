#!/usr/bin/env python3
"""Generate the Label Lag sample.

Constructed payments, not Stripe data. Fraud is overrepresented so the
lag is visible. Two training rules, both taken from the tradeoff in
Stripe's Scaling Radar post:

  - Fit now: train on every payment with the labels available today.
    Recent weeks help you learn new attacks, but many stolen-card
    disputes have not arrived yet, so some fraud is still labeled
    legitimate.

  - Wait for mature weeks: train only on payments that are old enough
    for most reports to have arrived. Labels are more trustworthy, at
    the cost of learning from a smaller, older window.

The distinctive late-reported pattern is a stolen card with years of
clean history on other merchants, making its first charge here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(2026)

FEATURE_META = [
    {"id": "card_age_days", "label": "Card age", "unit": "days",
     "raise_when": "A newer card usually raises risk."},
    {"id": "email_age_days", "label": "Email age", "unit": "days",
     "raise_when": "A newer email usually raises risk."},
    {"id": "ip_card_count", "label": "Cards seen on this IP", "unit": "cards",
     "raise_when": "Many cards on one IP usually raises risk."},
    {"id": "name_email_match", "label": "Name matches email", "unit": "yes/no",
     "raise_when": "A mismatch usually raises risk."},
    {"id": "amount_usd", "label": "Amount", "unit": "USD",
     "raise_when": "Very small test charges, or amounts far from a card's usual size, can raise risk."},
    {"id": "velocity_1h", "label": "Charges in the last hour", "unit": "charges",
     "raise_when": "A burst of charges usually raises risk."},
    {"id": "network_good_years", "label": "Clean history on other merchants", "unit": "years",
     "raise_when": "A long clean history usually lowers risk. That is also how late-reported fraud hides."},
    {"id": "first_time_at_merchant", "label": "First time at this merchant", "unit": "yes/no",
     "raise_when": "A first charge at this merchant, especially with a long history elsewhere, can raise risk."},
    {"id": "country_mismatch", "label": "Billing and IP country differ", "unit": "yes/no",
     "raise_when": "A country mismatch usually raises risk."},
    {"id": "throwaway_email", "label": "Throwaway-style email", "unit": "yes/no",
     "raise_when": "A disposable or patterned email usually raises risk."},
    {"id": "merchant_dispute_rate", "label": "Merchant's recent dispute rate", "unit": "percent",
     "raise_when": "A higher merchant dispute rate usually raises risk."},
]

MERCHANTS = [
    ("Northwell Books", 1.1), ("Copper & Pine", 0.8), ("Halcyon Tickets", 2.4),
    ("Kite & Co. Market", 0.6), ("Fieldglass Outdoor", 0.9), ("Velum Audio", 1.3),
    ("Paper Gnosis", 0.7), ("Red Maple Transit", 1.6), ("Lumen Courseware", 1.0),
    ("Orchard Lane Goods", 0.5), ("Bright Holt Apparel", 1.4), ("Sundial Coffee Roasters", 0.4),
]
FIRST = [
    "Amara", "Bennett", "Priya", "Julian", "Sasha", "Elena", "Marcus", "Leila",
    "Owen", "Nina", "Theo", "Iris", "Hugo", "Maya", "Colin", "Asha", "Felix",
    "Rae", "Jonah", "Mina", "Chris", "Ada", "Luca", "Noor", "Ivy", "Seth",
    "Dana", "Remy", "Zoe", "Anil",
]
LAST = [
    "Ellison", "Park", "Rahman", "Vega", "Cho", "Ibarra", "Nielsen", "Okoro",
    "Brooks", "Sato", "Klein", "DSouza", "Nguyen", "Berg", "Costa", "Shah",
    "Quinn", "Ito", "Larsen", "Moreau", "Khan", "Adler", "Bautista", "Fraser",
    "Yoon", "Patel", "Reed", "Gupta",
]
DOMAINS = ["mailleaf.com", "harborpost.net", "oakline.org", "bluewell.co", "paperhub.mail"]
THROWAWAY_DOMAINS = ["quickinbox.tmp", "boxdrop.xyz", "tempnote.net"]
CITIES = [
    ("Austin", "US", "198.51.100"), ("Chicago", "US", "203.0.113"),
    ("Portland", "US", "192.0.2"), ("Toronto", "CA", "198.51.100"),
    ("London", "GB", "203.0.113"), ("Berlin", "DE", "192.0.2"),
    ("Singapore", "SG", "198.51.100"), ("Mexico City", "MX", "203.0.113"),
]

TRAIN_DAYS = 70
EVAL_START = 71
EVAL_END = 77
HORIZON = 56
MATURE_WAIT = 42
OPERATING_FPR = 0.02

COL_NAMES = [
    "new_card", "new_email", "ip_cards", "name_mismatch", "amount_shape",
    "velocity", "network_history", "history_but_new_here", "country_mismatch",
    "throwaway_email", "merchant_dispute",
]
COL_LABELS = {
    "new_card": "Newer card",
    "new_email": "Newer email",
    "ip_cards": "Cards on this IP",
    "name_mismatch": "Name and email do not match",
    "amount_shape": "Unusual or test-sized amount",
    "velocity": "Charges in the last hour",
    "network_history": "Clean history on other merchants",
    "history_but_new_here": "Clean history elsewhere, first time here",
    "country_mismatch": "Billing and IP country differ",
    "throwaway_email": "Throwaway-style email",
    "merchant_dispute": "Merchant's recent dispute rate",
}


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def report_delay_days(kind, rng):
    if kind == "legit":
        return None
    if kind == "obvious":
        return int(np.clip(rng.lognormal(mean=1.45, sigma=0.50), 1, 18))
    return int(np.clip(rng.lognormal(mean=3.55, sigma=0.28), 21, 56))


def draw_payment(day, kind, rng):
    merchant, dispute_rate = MERCHANTS[int(rng.integers(0, len(MERCHANTS)))]
    city, country, ip_stem = CITIES[int(rng.integers(0, len(CITIES)))]
    first = FIRST[int(rng.integers(0, len(FIRST)))]
    last = LAST[int(rng.integers(0, len(LAST)))]

    if kind == "obvious":
        card_age = int(rng.integers(1, 18))
        email_age = int(rng.integers(0, 12))
        ip_card_count = int(rng.integers(4, 14))
        name_email_match = 0
        amount = float(rng.choice([1.0, 1.0, 2.0, 2.99, 5.0, 9.0]))
        velocity = int(rng.integers(3, 9))
        network_good_years = float(rng.choice([0.0, 0.0, 0.1, 0.2]))
        first_time = 1
        country_mismatch = 1 if rng.random() < 0.7 else 0
        throwaway = 1
        email = f"{first[0].lower()}{last.lower()}{int(rng.integers(10,99))}@{rng.choice(THROWAWAY_DOMAINS)}"
        billing_country = country if country_mismatch == 0 else rng.choice(
            [c[1] for c in CITIES if c[1] != country]
        )
    elif kind == "camouflage":
        card_age = int(rng.integers(500, 2800))
        email_age = int(rng.integers(300, 2400))
        ip_card_count = int(rng.integers(1, 3))
        name_email_match = 1 if rng.random() < 0.9 else 0
        amount = float(np.round(rng.uniform(48, 240), 2))
        velocity = int(rng.choice([1, 1, 2, 2, 3]))
        network_good_years = float(np.round(rng.uniform(2.4, 7.0), 1))
        first_time = 1 if rng.random() < 0.94 else 0
        country_mismatch = 1 if rng.random() < 0.28 else 0
        throwaway = 0
        email = f"{first.lower()}.{last.lower()}@{rng.choice(DOMAINS)}"
        billing_country = country if country_mismatch == 0 else rng.choice(
            [c[1] for c in CITIES if c[1] != country]
        )
    else:
        card_age = int(np.clip(rng.lognormal(5.9, 0.85), 30, 3200))
        email_age = int(np.clip(rng.lognormal(5.7, 0.95), 20, 3200))
        ip_card_count = int(np.clip(rng.poisson(1.1) + 1, 1, 7))
        name_email_match = 1 if rng.random() < 0.84 else 0
        amount = float(np.round(np.clip(rng.lognormal(3.7, 0.65), 8, 420), 2))
        velocity = int(np.clip(rng.poisson(0.35) + 1, 1, 5))
        network_good_years = float(np.round(np.clip(rng.lognormal(0.7, 0.75), 0, 8), 1))
        first_time = 1 if rng.random() < 0.10 else 0
        if first_time and rng.random() < 0.7:
            network_good_years = float(np.round(np.clip(network_good_years * 0.25, 0, 1.2), 1))
        country_mismatch = 1 if rng.random() < 0.07 else 0
        throwaway = 1 if rng.random() < 0.02 else 0
        if throwaway:
            email = f"{first[0].lower()}{last.lower()}{int(rng.integers(10,99))}@{rng.choice(THROWAWAY_DOMAINS)}"
        else:
            email = f"{first.lower()}.{last.lower()}@{rng.choice(DOMAINS)}"
        billing_country = country if country_mismatch == 0 else rng.choice(
            [c[1] for c in CITIES if c[1] != country]
        )

    local = int(rng.integers(10, 220))
    last4 = f"{int(rng.integers(1000, 9999))}"
    card_bin = f"{int(rng.integers(400000, 559999))}"
    return {
        "day": int(day), "kind": kind, "is_fraud": kind != "legit",
        "merchant": merchant, "amount_usd": amount,
        "card_age_days": int(card_age), "email_age_days": int(email_age),
        "ip_card_count": int(ip_card_count), "name_email_match": int(name_email_match),
        "velocity_1h": int(velocity), "network_good_years": float(network_good_years),
        "first_time_at_merchant": int(first_time), "country_mismatch": int(country_mismatch),
        "throwaway_email": int(throwaway), "merchant_dispute_rate": float(dispute_rate),
        "email": email, "card": f"{card_bin}******{last4}",
        "ip": f"{ip_stem}.{local}", "city": city, "ip_country": country,
        "billing_country": billing_country, "name": f"{first} {last}",
        "report_delay": report_delay_days(kind, rng),
    }


def design_matrix(rows):
    card_age = np.array([r["card_age_days"] for r in rows], dtype=float)
    email_age = np.array([r["email_age_days"] for r in rows], dtype=float)
    ip_cards = np.array([r["ip_card_count"] for r in rows], dtype=float)
    name_match = np.array([r["name_email_match"] for r in rows], dtype=float)
    amount = np.array([r["amount_usd"] for r in rows], dtype=float)
    velocity = np.array([r["velocity_1h"] for r in rows], dtype=float)
    history = np.array([r["network_good_years"] for r in rows], dtype=float)
    first_time = np.array([r["first_time_at_merchant"] for r in rows], dtype=float)
    mismatch = np.array([r["country_mismatch"] for r in rows], dtype=float)
    throwaway = np.array([r["throwaway_email"] for r in rows], dtype=float)
    dispute = np.array([r["merchant_dispute_rate"] for r in rows], dtype=float)
    new_card = np.log1p(np.maximum(0, 250 - card_age))
    new_email = np.log1p(np.maximum(0, 250 - email_age))
    amount_shape = (amount < 6).astype(float) + 0.25 * np.abs(np.log(amount / 55.0))
    history_but_new = np.log1p(history) * first_time
    return np.column_stack([
        new_card, new_email, np.log1p(ip_cards), 1.0 - name_match, amount_shape,
        np.log1p(velocity), np.log1p(history), history_but_new, mismatch,
        throwaway, dispute / 2.0,
    ])


def visible_label(row, as_of_day):
    if not row["is_fraud"]:
        return 0
    delay = row["report_delay"]
    if delay is None:
        return 0
    return 1 if row["day"] + delay <= as_of_day else 0


def fit_logit(X, y, l2=0.4, sample_weight=None):
    n, d = X.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    sw = sw / sw.mean()
    y_c = y - y.mean()
    w_init = np.zeros(d)
    for j in range(d):
        xj = X[:, j]
        denom = np.dot(xj - xj.mean(), xj - xj.mean()) + 1e-8
        w_init[j] = np.dot(xj - xj.mean(), y_c) / denom
    theta0 = np.concatenate([[0.0], w_init])

    def loss(theta):
        intercept, w = theta[0], theta[1:]
        p = np.clip(sigmoid(intercept + X @ w), 1e-8, 1 - 1e-8)
        nll = -np.sum(sw * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        return (nll / n) + l2 * np.sum(w**2) / n

    def grad(theta):
        intercept, w = theta[0], theta[1:]
        p = sigmoid(intercept + X @ w)
        err = sw * (p - y)
        return np.concatenate([[np.sum(err) / n], (X.T @ err + 2 * l2 * w) / n])

    res = minimize(loss, theta0, jac=grad, method="L-BFGS-B")
    if not res.success:
        print("warn: fit did not fully converge:", res.message)
    return res.x


def scores(theta, X):
    return sigmoid(theta[0] + X @ theta[1:])


def threshold_for_fpr(score, y, target_fpr):
    neg = score[y == 0]
    if len(neg) == 0:
        return 0.5
    order = np.sort(neg)[::-1]
    n = len(order)
    k = int(math.floor(target_fpr * n))
    if k <= 0:
        return float(order[0] + 1e-6)
    if k >= n:
        return 0.0
    return float(order[k - 1])


def recall_at_threshold(score, y, t):
    pos = y == 1
    if pos.sum() == 0:
        return None
    return float(np.mean(score[pos] >= t))


def fpr_at_threshold(score, y, t):
    neg = y == 0
    if neg.sum() == 0:
        return None
    return float(np.mean(score[neg] >= t))


def contributions(theta, x_row):
    intercept = float(theta[0])
    parts, logit = [], intercept
    for i, name in enumerate(COL_NAMES):
        add = float(theta[i + 1]) * float(x_row[i])
        logit += add
        parts.append({"id": name, "label": COL_LABELS[name], "contribution": round(add, 4)})
    parts.sort(key=lambda p: abs(p["contribution"]), reverse=True)
    return {"probability": float(sigmoid(logit)), "parts": parts}


def main():
    train, eval_rows = [], []
    for day in range(1, TRAIN_DAYS + 1):
        n = 30
        # Enough early camouflage that a model trained only on mature weeks
        # can still learn the "clean history elsewhere, first time here" pattern.
        # Later weeks add more of that pattern while it is still unlabeled.
        if day <= 24:
            n_cam, n_obs = 2, 1
        elif day <= 40:
            n_cam, n_obs = 1, 1
        else:
            n_cam, n_obs = 5, 1
        kinds = ["camouflage"] * n_cam + ["obvious"] * n_obs + ["legit"] * (n - n_cam - n_obs)
        RNG.shuffle(kinds)
        for kind in kinds:
            train.append(draw_payment(day, kind, RNG))

    for day in range(EVAL_START, EVAL_END + 1):
        n = 40
        n_cam, n_obs = 4, 2
        kinds = ["camouflage"] * n_cam + ["obvious"] * n_obs + ["legit"] * (n - n_cam - n_obs)
        RNG.shuffle(kinds)
        for kind in kinds:
            eval_rows.append(draw_payment(day, kind, RNG))

    train_as_of = EVAL_END
    X_train = design_matrix(train)
    means, scales = X_train.mean(axis=0), X_train.std(axis=0)
    scales[scales < 1e-6] = 1.0
    X_train_s = (X_train - means) / scales
    y_visible = np.array([visible_label(r, train_as_of) for r in train], dtype=float)
    y_true = np.array([1.0 if r["is_fraud"] else 0.0 for r in train])

    print("Feature corr with VISIBLE / TRUE:")
    for i, name in enumerate(COL_NAMES):
        c_v = np.corrcoef(X_train_s[:, i], y_visible)[0, 1]
        c_t = np.corrcoef(X_train_s[:, i], y_true)[0, 1]
        print(f"  {name:24s} vis={c_v:+.3f}  true={c_t:+.3f}")

    # Model "now": every payment, labels as of training day.
    # Recency weights make the immature weeks count more, which is the
    # "keep up with new attacks" choice in the post.
    recency = np.array([r["day"] for r in train], dtype=float)
    recency_w = 1.0 + 3.0 * (np.clip(recency - 40, 0, 30) / 30.0)
    theta_now = fit_logit(X_train_s, y_visible, l2=0.35, sample_weight=recency_w)

    # Model "wait": only weeks old enough that most reports should have arrived.
    mature_cut = train_as_of - MATURE_WAIT
    mature_idx = np.array([i for i, r in enumerate(train) if r["day"] <= mature_cut])
    print(f"Mature cutoff day {mature_cut}: {len(mature_idx)} payments, "
          f"visible fraud {int(y_visible[mature_idx].sum())}, "
          f"true fraud {int(y_true[mature_idx].sum())}")
    theta_wait = fit_logit(X_train_s[mature_idx], y_visible[mature_idx], l2=0.35)

    s_now_tr = scores(theta_now, X_train_s)
    s_wait_tr = scores(theta_wait, X_train_s)
        # Operating point: 2% FPR on the mature slice, using only the labels that
    # had arrived by training day (no peeking at disputes that come later).
    t_now = threshold_for_fpr(s_now_tr[mature_idx], y_visible[mature_idx], OPERATING_FPR)
    t_wait = threshold_for_fpr(s_wait_tr[mature_idx], y_visible[mature_idx], OPERATING_FPR)

    X_eval = design_matrix(eval_rows)
    X_eval_s = (X_eval - means) / scales
    s_now = scores(theta_now, X_eval_s)
    s_wait = scores(theta_wait, X_eval_s)

    y_end = np.array([1.0 if r["is_fraud"] else 0.0 for r in eval_rows])
    true_now = recall_at_threshold(s_now, y_end, t_now)
    true_wait = recall_at_threshold(s_wait, y_end, t_wait)

    series = []
    for lag in range(0, HORIZON + 1):
        as_of = EVAL_END + lag
        y = np.array([visible_label(r, as_of) for r in eval_rows], dtype=float)
        series.append({
            "day": lag, "as_of_day": as_of,
            "fraud_visible": int(y.sum()), "fraud_total": int(y_end.sum()),
            "reports_in": int(y.sum()),
            "now_apparent_recall": recall_at_threshold(s_now, y, t_now),
            "wait_apparent_recall": recall_at_threshold(s_wait, y, t_wait),
            "now_apparent_fpr": fpr_at_threshold(s_now, y, t_now),
            "wait_apparent_fpr": fpr_at_threshold(s_wait, y, t_wait),
            "now_true_recall": true_now, "wait_true_recall": true_wait,
        })

    out_payments = []
    for i, row in enumerate(eval_rows):
        item = dict(row)
        item["id"] = f"ch_{EVAL_START}_{i:04d}"
        item["score_now"] = round(float(s_now[i]), 4)
        item["score_wait"] = round(float(s_wait[i]), 4)
        item["flagged_now"] = bool(s_now[i] >= t_now)
        item["flagged_wait"] = bool(s_wait[i] >= t_wait)
        en, ew = contributions(theta_now, X_eval_s[i]), contributions(theta_wait, X_eval_s[i])
        item["explain_now"] = {"probability": round(en["probability"], 4), "parts": en["parts"][:8]}
        item["explain_wait"] = {"probability": round(ew["probability"], 4), "parts": ew["parts"][:8]}
        out_payments.append(item)

    interesting = [p for p in out_payments
                   if p["is_fraud"] and p["kind"] == "camouflage" and not p["flagged_now"]]
    related, used = [], 0
    other_merchants = [m[0] for m in MERCHANTS]
    for p in interesting:
        priors = []
        for k in range(2):
            prior_day = max(1, p["day"] - int(RNG.integers(12, 50)) - k * 9)
            merch = other_merchants[(used + k) % len(other_merchants)]
            if merch == p["merchant"]:
                merch = other_merchants[(used + k + 3) % len(other_merchants)]
            priors.append({
                "day": prior_day, "merchant": merch,
                "amount_usd": float(np.round(p["amount_usd"] * float(RNG.uniform(0.55, 1.15)), 2)),
                "email": p["email"], "card": p["card"],
                "ip": p["ip"] if k == 1 else f"{p['ip'].rsplit('.',1)[0]}.{int(RNG.integers(10,220))}",
                "is_fraud": False, "report_delay": None,
            })
        for row in priors:
            related.append({
                "id": f"ch_hist_{used:04d}", "day": int(row["day"]),
                "merchant": row["merchant"], "amount_usd": row["amount_usd"],
                "email": row["email"], "card": row["card"], "ip": row["ip"],
                "is_fraud": False, "report_delay": None,
                                "related_to": p["id"],
                "share": "card and IP" if row["ip"] == p["ip"] else "card",
            })
            used += 1

    def cam(pred):
        return sum(1 for p in out_payments if p["kind"] == "camouflage" and pred(p))

    summary = {
        "n_train": len(train), "n_eval": len(eval_rows),
        "train_fraud": int(y_true.sum()),
        "train_visible_fraud_at_fit": int(y_visible.sum()),
        "train_hidden_fraud_at_fit": int((y_true - y_visible).sum()),
        "n_mature_train": int(len(mature_idx)),
                "eval_fraud": int(y_end.sum()),
        "eval_legit": int((1 - y_end).sum()),
        "legit_flagged_now": int(((s_now >= t_now) & (y_end == 0)).sum()),
        "legit_flagged_wait": int(((s_wait >= t_wait) & (y_end == 0)).sum()),
        "eval_obvious": int(sum(1 for r in eval_rows if r["kind"] == "obvious")),
        "eval_camouflage": int(sum(1 for r in eval_rows if r["kind"] == "camouflage")),
        "mature_wait_days": MATURE_WAIT,
        "operating_fpr_target": OPERATING_FPR,
        "threshold_now": float(t_now), "threshold_wait": float(t_wait),
        "now_true_recall": true_now, "wait_true_recall": true_wait,
        "now_label_lag_gap": float(series[0]["now_apparent_recall"] - series[-1]["now_apparent_recall"]),
        "wait_label_lag_gap": float(series[0]["wait_apparent_recall"] - series[-1]["wait_apparent_recall"]),
        "now_day0_recall": series[0]["now_apparent_recall"],
        "wait_day0_recall": series[0]["wait_apparent_recall"],
        "now_day56_recall": series[-1]["now_apparent_recall"],
        "wait_day56_recall": series[-1]["wait_apparent_recall"],
        "train_as_of_day": train_as_of,
        "eval_start_day": EVAL_START, "eval_end_day": EVAL_END,
        "horizon_days": HORIZON, "mature_cutoff_day": int(mature_cut),
    }

    print("TRAIN", summary["n_train"], "visible", summary["train_visible_fraud_at_fit"],
          "hidden", summary["train_hidden_fraud_at_fit"], "true", summary["train_fraud"])
    print("EVAL", summary["n_eval"], "fraud", summary["eval_fraud"],
          "obvious", summary["eval_obvious"], "camouflage", summary["eval_camouflage"])
    print("THRESH now", round(t_now, 4), "wait", round(t_wait, 4))
    for d in (0, 7, 14, 21, 28, 42, 56):
        s = series[d]
        print(f"DAY {d:2d}  now {s['now_apparent_recall']:.3f}  wait {s['wait_apparent_recall']:.3f}  "
              f"visible {s['fraud_visible']}/{s['fraud_total']}")
    print("TRUE    now", true_now, "wait", true_wait)
    print("GAP     now", summary["now_label_lag_gap"], "wait", summary["wait_label_lag_gap"])
    print("Weights NOW", {k: round(float(w), 3) for k, w in zip(COL_NAMES, theta_now[1:])})
    print("Weights WAIT", {k: round(float(w), 3) for k, w in zip(COL_NAMES, theta_wait[1:])})
    print("Camouflage caught now / wait:", cam(lambda p: p["flagged_now"]), cam(lambda p: p["flagged_wait"]))
    print("Obvious caught now / wait:",
          sum(1 for p in out_payments if p["kind"] == "obvious" and p["flagged_now"]),
          sum(1 for p in out_payments if p["kind"] == "obvious" and p["flagged_wait"]))
    print("Camouflage missed now, caught wait:",
          cam(lambda p: (not p["flagged_now"]) and p["flagged_wait"]))
    print("Legit flagged now / wait:",
          sum(1 for p in out_payments if not p["is_fraud"] and p["flagged_now"]),
          sum(1 for p in out_payments if not p["is_fraud"] and p["flagged_wait"]))

    payload = {
        "disclaimer": (
            "This is a constructed sample, not Stripe data. Fraud is overrepresented "
            "so the lag is visible in a short demo. In live card payments, fraud is "
            "closer to 1 in 1,000."
        ),
        "source": {
            "title": "Scaling Radar: Fighting fraud with agent-assisted machine learning",
            "authors": ["Ryan Drapeau", "Sai Samant"],
            "date": "2026-10-19",
            "url": "https://stripe.dev/blog/scaling-radar-fighting-fraud-with-agent-assisted-machine-learning",
            "related": {
                "title": "How we built it: Stripe Radar",
                "url": "https://stripe.dev/blog/how-we-built-it-stripe-radar",
            },
        },
        "features": FEATURE_META,
        "feature_columns": [{"id": k, "label": COL_LABELS[k]} for k in COL_NAMES],
        "models": {
            "now": {
                "id": "now",
                "name": "Fit now, with the labels you have",
                "short": "Fit now",
                "blurb": "Train on every payment in the 10-week window, using only the fraud reports that had arrived by training day. Recent weeks count more, so the model can chase new attacks, and also so immature labels can do more damage.",
                "threshold": float(t_now),
                "weights": {k: float(w) for k, w in zip(COL_NAMES, theta_now[1:])},
                "intercept": float(theta_now[0]),
            },
            "wait": {
                "id": "wait",
                "name": "Train only on mature weeks",
                "short": "Wait 6 weeks",
                "blurb": "Drop the most recent 6 weeks and train only on payments that are old enough for most disputes to have arrived. Labels are more trustworthy. The window is smaller and a little stale.",
                "threshold": float(t_wait),
                "weights": {k: float(w) for k, w in zip(COL_NAMES, theta_wait[1:])},
                "intercept": float(theta_wait[0]),
            },
        },
        "summary": summary,
        "series": series,
        "payments": out_payments,
        "related": related,
    }

    # Keep full explanations only on fraud and on flagged legitimate payments.
    keep_explain = set()
    for pmt in out_payments:
        if pmt["is_fraud"] or pmt["flagged_now"] or pmt["flagged_wait"]:
            keep_explain.add(pmt["id"])
        else:
            pmt.pop("explain_now", None)
            pmt.pop("explain_wait", None)

    out = Path(__file__).resolve().parent / "data" / "sample.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print("Wrote", out, "bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
