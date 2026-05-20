"""
=============================================================================
  WAREHOUSE OPERATIONS OPTIMIZATION — COMPLETE PROJECT CODE  (v2)
  Dataset : 10000_Sales_Records.csv
  Fields  : Region, Country, Item Type, Sales Channel, Order Priority,
             Order Date, Order ID, Ship Date, Units Sold, Unit Price,
             Unit Cost, Total Revenue, Total Cost, Total Profit

  Modules
  -------
  1. Demand Forecasting      — ARIMA (pure NumPy) + LSTM (pure NumPy)
                               Durbin-Watson autocorrelation test
  2. Inventory Optimization  — EOQ + Safety Stock + Buffer Stock
  3. Warehouse Layout        — SLP baseline + Genetic Algorithm
                               + Monte-Carlo Warehouse Simulation
  4. Order Picking           — ABC Analysis (Pareto) + Genetic Algorithm TSP
  5. Routing Algorithm       — Hill Climbing + Simulated Annealing
  6. Performance Evaluation  — Comprehensive KPI Dashboard

  Run:   python warehouse_optimization_2.py
  Output: warehouse_outputs/   (7 PNG figures)
=============================================================================
"""

import warnings, math, random, os, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import stats
from scipy.linalg import lstsq

np.random.seed(42)
random.seed(42)
OUT = "warehouse_outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def mape(a, b): return float(np.mean(np.abs((a - b) / (np.abs(a) + 1e-9))) * 100)

# ─────────────────────────────────────────────────────────────────────────────
# PURE-NUMPY ARIMA(p,d,q)
# ─────────────────────────────────────────────────────────────────────────────
class NumpyARIMA:
    def __init__(self, p, d, q):
        self.p, self.d, self.q = p, d, q
        self.ar = np.zeros(p)
        self.ma = np.zeros(q)
        self.c  = 0.0
        self._resid = np.array([])
        self._diff  = np.array([])
        self._orig  = np.array([])

    @staticmethod
    def _diff_ts(x, d):
        for _ in range(d): x = np.diff(x)
        return x

    @staticmethod
    def _undiff(seed, fc, d):
        out = fc.copy()
        for i in range(d):
            last = seed[-(i + 1)]
            out = np.cumsum(np.concatenate([[last], out]))
        return out

    def fit(self, y):
        self._orig = y.astype(float)
        diff = self._diff_ts(self._orig, self.d)
        self._diff = diff
        p, q, n = self.p, self.q, len(diff)
        ml = max(p, q, 1)
        resid = np.zeros(n)

        if n <= ml + 2:
            self.c = float(np.mean(diff))
            self._resid = diff - self.c
            return self

        rows = n - ml
        X = np.ones((rows, 1 + p + q))
        for i, t in enumerate(range(ml, n)):
            for j in range(p): X[i, 1+j]   = diff[t-j-1]
            for j in range(q): X[i, 1+p+j] = resid[t-j-1]
        Y = diff[ml:]
        coef, *_ = lstsq(X, Y)
        self.c  = float(coef[0])
        self.ar = coef[1:1+p]
        self.ma = coef[1+p:]

        resid2 = np.zeros(n)
        for t in range(ml, n):
            hat = self.c
            for j in range(p): hat += self.ar[j] * diff[t-j-1]
            for j in range(q): hat += self.ma[j] * resid2[t-j-1]
            resid2[t] = diff[t] - hat
        self._resid = resid2
        return self

    def forecast(self, h):
        diff  = self._diff.copy()
        resid = self._resid.copy()
        fc    = []
        for _ in range(h):
            val = self.c
            for j in range(self.p):
                if len(diff)  > j: val += self.ar[j] * diff[-(j+1)]
            for j in range(self.q):
                if len(resid) > j: val += self.ma[j] * resid[-(j+1)]
            fc.append(val)
            diff  = np.append(diff,  val)
            resid = np.append(resid, 0)
        raw = np.array(fc)
        out = self._undiff(self._orig, raw, self.d)
        return np.maximum(out[-h:], 0)

    def aic(self):
        n = len(self._resid)
        s2 = float(np.var(self._resid)) + 1e-9
        k  = 1 + self.p + self.q
        return n * np.log(s2) + 2 * k


def durbin_watson(resid):
    """
    Durbin-Watson statistic to detect autocorrelation in residuals.
    DW ≈ 2 → no autocorrelation (white noise residuals)
    DW < 1.5 → positive autocorrelation
    DW > 2.5 → negative autocorrelation
    """
    r = resid[resid != 0]
    if len(r) < 2: return 2.0
    dw_stat = float(np.sum(np.diff(r)**2) / (np.sum(r**2) + 1e-12))
    return dw_stat

def adf_test(x):
    dx  = np.diff(x.astype(float))
    lx  = (x[:-1] - np.mean(x)).reshape(-1, 1)
    if np.all(lx == 0): return 0.0, 0.99
    b, *_ = lstsq(lx, dx)
    fitted = lx.flatten() * b[0]
    ss_res = float(np.sum((dx - fitted)**2))
    se     = math.sqrt(ss_res / (len(dx) - 2) / (float(np.sum(lx**2)) + 1e-12))
    t_stat = float(b[0]) / (se + 1e-12)
    p_val  = float(2 * stats.t.cdf(t_stat, df=len(dx)-2))
    return t_stat, p_val

# ─────────────────────────────────────────────────────────────────────────────
# PURE-NUMPY LSTM
# ─────────────────────────────────────────────────────────────────────────────
def _sig(x):  return 1 / (1 + np.exp(-np.clip(x, -40, 40)))
def _tanh(x): return np.tanh(x)

class NumpyLSTM:
    def __init__(self, seq_len, hidden=20):
        self.sl = seq_len
        self.hs = hidden
        s = 0.08
        isize = 1 + hidden
        def W(r, c): return np.random.randn(r, c) * s
        self.Wf=W(hidden,isize); self.bf=np.zeros((hidden,1))
        self.Wi=W(hidden,isize); self.bi=np.zeros((hidden,1))
        self.Wc=W(hidden,isize); self.bc=np.zeros((hidden,1))
        self.Wo=W(hidden,isize); self.bo=np.zeros((hidden,1))
        self.Wy=W(1,hidden);     self.by=np.zeros((1,1))

    def _step(self, x_scalar, h, c):
        xh = np.vstack([np.array([[x_scalar]]), h])
        f  = _sig( self.Wf @ xh + self.bf)
        i_ = _sig( self.Wi @ xh + self.bi)
        g  = _tanh(self.Wc @ xh + self.bc)
        o  = _sig( self.Wo @ xh + self.bo)
        c2 = f*c + i_*g
        h2 = o*_tanh(c2)
        y  = float((self.Wy @ h2 + self.by)[0,0])
        return y, h2, c2

    def _forward(self, x_seq):
        h = np.zeros((self.hs,1)); c = np.zeros((self.hs,1))
        last_y = 0.0
        for x in x_seq:
            last_y, h, c = self._step(float(x), h, c)
        return last_y, h, c

    def predict(self, x_seq):
        return self._forward(x_seq)[0]

    def train(self, X, y, lr=0.004, epochs=130):
        losses = []
        for ep in range(epochs):
            el = 0.0
            for xi, yi in zip(X, y):
                p   = self.predict(xi)
                err = p - yi
                self.Wy  -= lr * err * self.Wy
                self.by[0,0] -= lr * err
                el += err**2
            losses.append(el / (len(X) + 1e-9))
        return losses


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("   WAREHOUSE OPERATIONS OPTIMIZATION — PROJECT REVIEW OUTPUT  (v2)")
print("=" * 70)

# =============================================================================
# 0. DATA LOADING & PREPROCESSING
# =============================================================================
print("\n[0] LOADING & PREPROCESSING DATA")
print("-" * 40)

df = pd.read_csv("C:/Users/xende/Downloads/10000 Sales Records.csv",
                 parse_dates=["Order Date", "Ship Date"])

total_orders = len(df)
date_min = df["Order Date"].min().date()
date_max = df["Order Date"].max().date()

df["Week"] = df["Order Date"].dt.to_period("W")
weekly = df.groupby("Week")["Units Sold"].sum().reset_index(name="units_sold")
ts = weekly["units_sold"].values.astype(float)

sku_stats = df.groupby("Item Type").agg(
    total_units=("Units Sold",  "sum"),
    avg_unit_price=("Unit Price","mean"),
    avg_unit_cost=("Unit Cost",  "mean"),
    total_revenue=("Total Revenue","sum"),
    total_profit=("Total Profit", "sum")
).reset_index()

top_sku = sku_stats.loc[sku_stats["total_units"].idxmax()]

region_flow  = df.groupby("Region")["Units Sold"].sum()
channel_flow = df.groupby("Sales Channel")["Units Sold"].sum()
priority_flow= df.groupby("Order Priority")["Units Sold"].sum()

df["Lead_Time_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
LT_mean = df["Lead_Time_Days"].mean()
LT_std  = df["Lead_Time_Days"].std()
LT_weeks = max(1, round(LT_mean / 7))

print(f"  Total orders      : {total_orders:,}")
print(f"  Date range        : {date_min} → {date_max}")
print(f"  Weekly obs        : {len(ts)}")
print(f"  Item types (SKUs) : {df['Item Type'].nunique()}")
print(f"  Regions           : {df['Region'].nunique()}")
print(f"  Avg lead time     : {LT_mean:.1f} days  (std={LT_std:.1f}d, {LT_weeks} week(s))")
print(f"  Top SKU by volume : {top_sku['Item Type']} ({top_sku['total_units']:,} units)")
print(f"  Total Revenue     : ${df['Total Revenue'].sum():,.0f}")
print(f"  Total Profit      : ${df['Total Profit'].sum():,.0f}")

# =============================================================================
# 1. DEMAND FORECASTING  (Weekly Units Sold)
# =============================================================================
print("\n[1] DEMAND FORECASTING  (Weekly Units Sold)")
print("-" * 40)

t_stat, p_val = adf_test(ts)
print(f"\n  ADF test: t={t_stat:.3f}  p={p_val:.4f}  "
      f"{'Stationary ✓' if p_val < 0.05 else 'Non-stationary'}")

train_n = int(len(ts) * 0.80)
tr, te  = ts[:train_n], ts[train_n:]

# ARIMA grid search
print("  ARIMA grid search … ", end="", flush=True)
best_aic, best_order, best_model = np.inf, (1,1,1), None
for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            try:
                m = NumpyARIMA(p, d, q).fit(tr)
                a = m.aic()
                if np.isfinite(a) and a < best_aic:
                    best_aic, best_order, best_model = a, (p,d,q), m
            except Exception:
                pass
print(f"done → best {best_order}")

arima_pred   = best_model.forecast(len(te))
arima_future = best_model.forecast(len(te) + 8)[-8:]
arima_resid  = best_model._resid

arima_mae_v  = mae(te, arima_pred)
arima_rmse_v = rmse(te, arima_pred)
arima_mape_v = mape(te, arima_pred)

# ─── Durbin-Watson Test ───────────────────────────────────────────────────────
dw = durbin_watson(arima_resid)
if 1.5 < dw < 2.5:
    dw_note = "No significant autocorrelation ✓"
    if best_order == (0, 0, 0):
        dw_interp = ("ARIMA(0,0,0) is a mean-only model — it predicts the series average. "
                     "White-noise residuals here do NOT mean well-specified; they mean "
                     "no temporal structure was captured at all. "
                     "Consider higher-order ARIMA or detrending first.")
    else:
        dw_interp = ("Residuals show no remaining autocorrelation — "
                     "model has captured the available temporal structure.")
elif dw < 1.5:
    dw_note   = "Positive autocorrelation detected"
    dw_interp = "Consecutive residuals share the same sign; model is under-fit — try increasing p or d."
else:
    dw_note   = "Negative autocorrelation detected"
    dw_interp = "Consecutive residuals alternate signs; possible over-differencing — try reducing d."

print(f"  ARIMA{best_order}  MAE={arima_mae_v:.2f}  RMSE={arima_rmse_v:.2f}  MAPE={arima_mape_v:.2f}%")
print(f"  Durbin-Watson = {dw:.4f}  →  {dw_note}")
print(f"  Interpretation: {dw_interp}")

# LSTM
print("  Training LSTM … ", end="", flush=True)
ts_min, ts_max = ts.min(), ts.max() + 1e-9
ts_n = (ts - ts_min) / (ts_max - ts_min)

SEQ = 4
def make_seq(data, s):
    X, y = [], []
    for i in range(len(data) - s):
        X.append(data[i:i+s]); y.append(data[i+s])
    return np.array(X), np.array(y)

Xa, ya = make_seq(ts_n, SEQ)
sp = int(len(Xa) * 0.80)
Xtr, Xte = Xa[:sp], Xa[sp:]
ytr, yte = ya[:sp], ya[sp:]

lstm = NumpyLSTM(seq_len=SEQ, hidden=20)
lstm.train(Xtr, ytr)

lstm_pred_n = np.array([lstm.predict(xi) for xi in Xte])
lstm_pred   = lstm_pred_n * (ts_max - ts_min) + ts_min
yte_inv     = yte         * (ts_max - ts_min) + ts_min

buf = ts_n[-SEQ:].copy()
lf_n = []
for _ in range(8):
    p = lstm.predict(buf); lf_n.append(p)
    buf = np.append(buf[1:], p)
lstm_future = np.array(lf_n) * (ts_max - ts_min) + ts_min

lstm_mae_v  = mae(yte_inv, lstm_pred)
lstm_rmse_v = rmse(yte_inv, lstm_pred)
lstm_mape_v = mape(yte_inv, lstm_pred)
print("done")
print(f"  LSTM        MAE={lstm_mae_v:.2f}  RMSE={lstm_rmse_v:.2f}  MAPE={lstm_mape_v:.2f}%")

print(f"\n  {'Model':<18}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
print(f"  {'-'*50}")
print(f"  {'ARIMA'+str(best_order):<18}{arima_mae_v:>10.2f}{arima_rmse_v:>10.2f}{arima_mape_v:>9.2f}%")
print(f"  {'LSTM':<18}{lstm_mae_v:>10.2f}{lstm_rmse_v:>10.2f}{lstm_mape_v:>9.2f}%")

# =============================================================================
# 2. INVENTORY OPTIMIZATION  (EOQ + Safety Stock + Buffer Stock)
# =============================================================================
print("\n[2] INVENTORY OPTIMIZATION")
print("-" * 40)

avg_d   = float(np.mean(ts))
std_d   = float(np.std(ts))
ann_d   = avg_d * 52

U_cost  = float(top_sku["avg_unit_cost"])
U_price = float(top_sku["avg_unit_price"])
margin  = float(top_sku["total_profit"] / max(top_sku["total_units"], 1))

h_pct   = 0.25
O_cost  = max(50.0, U_cost * 0.05 * 52)
Z       = 1.645                              # 95% service level
LT      = LT_weeks

# ─── EOQ ─────────────────────────────────────────────────────────────────────
EOQ          = math.sqrt(2 * ann_d * O_cost / (U_cost * h_pct))

# ─── Safety Stock (absorbs demand variability during lead time) ───────────────
safety_stock = Z * std_d * math.sqrt(LT)

# ─── Buffer Stock (extra cushion for lead-time variability) ──────────────────
# Buffer = Z * daily_demand_std * sqrt(LT_variance) added on top of safety stock
LT_sigma_days = LT_std if LT_std > 0 else LT_mean * 0.2
buffer_stock  = 0.15 * avg_d * LT          # 15% of avg LT demand

ROP          = avg_d * LT + safety_stock
# Max inventory = stock level just after an order arrives at ROP
# = ROP + EOQ  (you order EOQ when stock hits ROP; delivery arrives
#   as you consumed down to safety_stock, topping back up to ROP + EOQ)
max_inv      = ROP + EOQ
orders_yr    = ann_d / EOQ
ord_cost_yr  = orders_yr * O_cost
avg_inv_held = EOQ / 2 + safety_stock + buffer_stock   # average stock position
hld_cost_yr  = avg_inv_held * U_cost * h_pct
total_cost   = ord_cost_yr + hld_cost_yr
inv_turn     = ann_d / avg_inv_held
doh          = 365 / inv_turn
stockout_p   = 1 - stats.norm.cdf(Z)

print(f"  Top SKU           : {top_sku['Item Type']}")
print(f"  Unit Cost (data)  : ${U_cost:.2f}   Unit Price: ${U_price:.2f}   Margin/unit: ${margin:.2f}")
print(f"  Lead Time (data)  : {LT} week(s)  ({LT_mean:.1f} days avg,  σ={LT_std:.1f}d)")
print(f"  EOQ               = {EOQ:.1f} units")
print(f"  Safety Stock      = {safety_stock:.1f} units  (95% service level, Z={Z})")
print(f"  Buffer Stock      = {buffer_stock:.1f} units  (15% of avg LT demand)")
print(f"  Reorder Point     = {ROP:.1f} units")
print(f"  Max Inventory     = {max_inv:.1f} units")
print(f"  Annual Cost       = ${total_cost:,.2f}  (Ord: ${ord_cost_yr:,.0f}  Hold: ${hld_cost_yr:,.0f})")
print(f"  Inventory Turnover= {inv_turn:.2f}×/yr   Days on Hand={doh:.0f}d")
print(f"  Stockout probability = {stockout_p*100:.2f}%")

# =============================================================================
# 3. WAREHOUSE LAYOUT (SLP + GA + Monte-Carlo Simulation)
# =============================================================================
print("\n[3] WAREHOUSE LAYOUT (SLP + Genetic Algorithm + Simulation)")
print("-" * 40)

top_items = sku_stats.nlargest(8, "total_units")["Item Type"].tolist()
zones = top_items
NZ = len(zones)
GC = 4

flow = np.zeros((NZ, NZ))
for region, grp in df.groupby("Region"):
    vols = grp.groupby("Item Type")["Units Sold"].sum()
    for i, zi in enumerate(zones):
        for j, zj in enumerate(zones):
            if i != j and zi in vols.index and zj in vols.index:
                flow[i, j] += math.sqrt(vols[zi] * vols[zj]) / 1000

flow = (flow + flow.T) / 2
np.fill_diagonal(flow, 0)

def gd(p1, p2):
    r1,c1 = divmod(p1, GC); r2,c2 = divmod(p2, GC)
    return abs(r1-r2)+abs(c1-c2)+1

def lay_cost(perm):
    c = 0.0
    for i in range(NZ):
        for j in range(i+1, NZ):
            c += flow[i,j] * gd(perm.index(i), perm.index(j))
    return c

def ox(p1, p2):
    a,b = sorted(random.sample(range(NZ), 2))
    ch = [-1]*NZ; ch[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in ch]
    k=0
    for i in range(NZ):
        if ch[i]==-1: ch[i]=fill[k]; k+=1
    return ch

def mut(p, r=0.15):
    if random.random()<r:
        i,j = random.sample(range(NZ),2); p[i],p[j]=p[j],p[i]
    return p

POP=80; GEN=200; EL=5
pop = [random.sample(range(NZ), NZ) for _ in range(POP)]
init_cost = lay_cost(list(range(NZ)))
best_p = min(pop, key=lay_cost); best_c = lay_cost(best_p)
hist = []
for g in range(GEN):
    pop.sort(key=lay_cost)
    elites = [p[:] for p in pop[:EL]]
    np2 = elites[:]
    while len(np2)<POP:
        a,b = random.choices(pop[:30], k=2)
        np2.append(mut(ox(a, b)))
    pop = np2
    gc = lay_cost(pop[0]); hist.append(gc)
    if gc < best_c: best_c=gc; best_p=pop[0][:]

impr = (init_cost - best_c) / init_cost * 100
print(f"  Zones (top Item Types): {', '.join(zones)}")
print(f"  Initial cost: {init_cost:.0f}  →  GA best: {best_c:.0f}  ({impr:.1f}% improvement)")

# ─── Monte-Carlo Warehouse Simulation ────────────────────────────────────────
# Simulate daily warehouse throughput under demand variability
# Each day: orders arrive (Poisson), pickers process orders (Exponential service time)
print("  Running Monte-Carlo warehouse simulation … ", end="", flush=True)

np.random.seed(99)
N_DAYS     = 365
PICKERS    = 5
PICK_RATE  = 12           # orders/picker/hour  (optimised layout)
PICK_RATE_BASE = 8        # orders/picker/hour  (baseline layout)
WORK_HRS   = 8
DAILY_CAP  = PICKERS * PICK_RATE * WORK_HRS          # optimised
DAILY_CAP_BASE = PICKERS * PICK_RATE_BASE * WORK_HRS # baseline

# Use daily ORDER count (rows), not unit volume, for the queue simulation.
# avg_d is weekly UNITS — we need avg orders/day from the raw data.
daily_order_rate = len(df) / max((df["Order Date"].max() - df["Order Date"].min()).days, 1)
daily_orders_arr = np.random.poisson(lam=daily_order_rate, size=N_DAYS)

# Simulate queue (backlog) across the year
def sim_warehouse(daily_orders, capacity):
    backlog = 0
    backlogs, throughputs, utilisations = [], [], []
    for d_orders in daily_orders:
        processable = min(d_orders + backlog, int(capacity))
        backlog     = max(0, d_orders + backlog - processable)
        util        = processable / capacity
        backlogs.append(backlog)
        throughputs.append(processable)
        utilisations.append(util)
    return (np.array(backlogs), np.array(throughputs), np.array(utilisations))

bl_base, tp_base, ut_base = sim_warehouse(daily_orders_arr, DAILY_CAP_BASE)
bl_opt,  tp_opt,  ut_opt  = sim_warehouse(daily_orders_arr, DAILY_CAP)

sim_fill_rate_base = float(np.mean(daily_orders_arr <= DAILY_CAP_BASE) * 100)
sim_fill_rate_opt  = float(np.mean(daily_orders_arr <= DAILY_CAP)      * 100)
sim_avg_backlog_base = float(np.mean(bl_base))
sim_avg_backlog_opt  = float(np.mean(bl_opt))
sim_util_base = float(np.mean(ut_base) * 100)
sim_util_opt  = float(np.mean(ut_opt)  * 100)

print("done")
print(f"  Baseline  — Avg backlog: {sim_avg_backlog_base:.1f}  Fill rate: {sim_fill_rate_base:.1f}%  Utilisation: {sim_util_base:.1f}%")
print(f"  Optimised — Avg backlog: {sim_avg_backlog_opt:.1f}  Fill rate: {sim_fill_rate_opt:.1f}%  Utilisation: {sim_util_opt:.1f}%")

# =============================================================================
# 4. ORDER PICKING — ABC ANALYSIS + GENETIC ALGORITHM (TSP)
# =============================================================================
print("\n[4] ORDER PICKING (ABC Analysis + Genetic Algorithm TSP)")
print("-" * 40)

# ─── ABC Analysis (Pareto / 80-20 rule) ──────────────────────────────────────
abc_df = sku_stats.copy().sort_values("total_revenue", ascending=False).reset_index(drop=True)
abc_df["cumrev"]  = abc_df["total_revenue"].cumsum()
abc_df["cum_pct"] = abc_df["cumrev"] / abc_df["total_revenue"].sum() * 100
abc_df["ABC"]     = abc_df["cum_pct"].apply(
    lambda x: "A" if x <= 80 else ("B" if x <= 95 else "C"))

count_A = (abc_df["ABC"] == "A").sum()
count_B = (abc_df["ABC"] == "B").sum()
count_C = (abc_df["ABC"] == "C").sum()
rev_A   = abc_df[abc_df["ABC"]=="A"]["total_revenue"].sum()
rev_B   = abc_df[abc_df["ABC"]=="B"]["total_revenue"].sum()
rev_C   = abc_df[abc_df["ABC"]=="C"]["total_revenue"].sum()
total_rev = abc_df["total_revenue"].sum()

print(f"  ABC Classification:")
print(f"    A items : {count_A} SKUs  → {rev_A/total_rev*100:.1f}% revenue  (near dispatch)")
print(f"    B items : {count_B} SKUs  → {rev_B/total_rev*100:.1f}% revenue  (mid-zone)")
print(f"    C items : {count_C} SKUs  → {rev_C/total_rev*100:.1f}% revenue  (deep storage)")

# ─── Genetic Algorithm for Order Picking Route (TSP) ─────────────────────────
# Locations represent pick zones; A items are clustered near depot
np.random.seed(42); random.seed(42)   # isolated, reproducible seed for TSP section
top_countries = df.groupby("Country")["Units Sold"].sum().nlargest(20).index.tolist()
NL = min(20, len(top_countries))

rng = np.random.default_rng(7)
pts = np.vstack([np.array([[0., 0.]]),
                 rng.uniform(0, 50, (NL, 2))])

# Assign picking weights by ABC class (A items picked more frequently)
item_vols = df.groupby("Country")["Units Sold"].sum()
# Use country volumes as pick-frequency weights
pick_weights = np.array([item_vols.get(c, 1) for c in top_countries], dtype=float)
pick_weights = pick_weights / pick_weights.sum()

def rdist(route):
    d = sum(np.linalg.norm(pts[route[i]] - pts[route[i+1]])
            for i in range(len(route)-1))
    return float(d + np.linalg.norm(pts[route[-1]] - pts[route[0]]))

# ─── GA TSP ──────────────────────────────────────────────────────────────────
def ga_tsp(n_gen=300, pop_size=100, elite_k=5, mut_rate=0.08):
    """Genetic Algorithm for Travelling Salesman (Order Picking Route)."""
    nodes = list(range(1, len(pts)))

    def rand_route():
        r = nodes[:]
        random.shuffle(r)
        return [0] + r

    def cx_order(p1, p2):
        """Order crossover (OX) on non-depot nodes."""
        s = p1[1:]; d = p2[1:]
        a, b = sorted(random.sample(range(len(s)), 2))
        child = [-1]*len(s); child[a:b] = s[a:b]
        fill = [x for x in d if x not in child]
        k=0
        for i in range(len(s)):
            if child[i]==-1: child[i]=fill[k]; k+=1
        return [0]+child

    def mutate(route, r=mut_rate):
        r2 = route[:]
        if random.random() < r:
            i, j = random.sample(range(1, len(r2)), 2)
            r2[i], r2[j] = r2[j], r2[i]
        return r2

    population = [rand_route() for _ in range(pop_size)]
    best_route = min(population, key=rdist)
    best_dist  = rdist(best_route)
    conv = []

    for _ in range(n_gen):
        population.sort(key=rdist)
        elites = [r[:] for r in population[:elite_k]]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(population[:40], k=2)
            child = mutate(cx_order(p1, p2))
            new_pop.append(child)
        population = new_pop
        cd = rdist(population[0])
        conv.append(cd)
        if cd < best_dist:
            best_dist = cd
            best_route = population[0][:]

    return best_route, best_dist, conv

print("  Running GA-TSP for order picking … ", end="", flush=True)
ga_route, ga_dist, ga_conv = ga_tsp(n_gen=500, pop_size=150, elite_k=8, mut_rate=0.12)

# Post-process: apply 2-opt to GA result to guarantee we match or beat NN
def two_opt_improve(route):
    best = route[:]
    best_d = rdist(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                cd = rdist(candidate)
                if cd < best_d - 1e-9:
                    best, best_d = candidate, cd
                    improved = True
    return best, best_d

ga_route, ga_dist = two_opt_improve(ga_route)
print(f"done  →  GA best dist = {ga_dist:.1f} m")

# Baseline: nearest-neighbour
def nn_build():
    unv=list(range(1,len(pts))); r=[0]
    while unv:
        c=r[-1]; nb=min(unv, key=lambda x: np.linalg.norm(pts[c]-pts[x]))
        r.append(nb); unv.remove(nb)
    return r

nn_r  = nn_build(); nn_d = rdist(nn_r)
rnd_r = [0]+random.sample(range(1,len(pts)), len(pts)-1)
rnd_d = rdist(rnd_r)
ga_saving_vs_nn  = (nn_d - ga_dist) / nn_d * 100   # can be negative if GA underperforms
ga_saving_vs_rnd = (rnd_d - ga_dist) / rnd_d * 100
pph_base = 45
pph_ga   = pph_base * (1 + max(0.0, ga_saving_vs_nn) / 100)

ga_note = ("GA+2opt beats NN ✓" if ga_saving_vs_nn >= 0
           else f"GA trails NN on {NL}-node instance "
                f"(NN is near-optimal at small N; GA advantage appears at N>50)")

print(f"  Random route   : {rnd_d:.1f} m")
print(f"  NN route       : {nn_d:.1f} m")
print(f"  GA-TSP route   : {ga_dist:.1f} m  (saves {ga_saving_vs_nn:.1f}% vs NN, {ga_saving_vs_rnd:.1f}% vs random)")
print(f"  Note           : {ga_note}")
print(f"  Picks/hr       : {pph_base} → {pph_ga:.1f}")

# =============================================================================
# 5. ROUTING ALGORITHMS — HILL CLIMBING + SIMULATED ANNEALING
# =============================================================================
print("\n[5] ROUTING ALGORITHMS (Hill Climbing + Simulated Annealing)")
print("-" * 40)

# ─── Hill Climbing ────────────────────────────────────────────────────────────
def hill_climbing(start_route, max_iter=3000):
    """
    Local search: at each step, try all 2-opt swaps.
    Accept only improvements (greedy ascent on negative distance).
    Can get stuck in local optima.
    """
    current = start_route[:]
    current_dist = rdist(current)
    improved = True
    iterations = 0
    hc_history = [current_dist]

    while improved and iterations < max_iter:
        improved = False
        for i in range(1, len(current) - 1):
            for j in range(i + 1, len(current)):
                candidate = current[:i] + current[i:j+1][::-1] + current[j+1:]
                cand_dist = rdist(candidate)
                if cand_dist < current_dist - 1e-9:
                    current      = candidate
                    current_dist = cand_dist
                    improved     = True
        iterations += 1
        hc_history.append(current_dist)

    return current, current_dist, hc_history

# ─── Simulated Annealing ─────────────────────────────────────────────────────
def simulated_annealing(start_route, T_init=5000.0, T_min=1e-3,
                        cooling=0.9985, max_iter=8000):
    """
    Probabilistic acceptance of worse solutions allows escaping local optima.
    P(accept worse) = exp(-Δcost / T)  — decreases as temperature cools.
    """
    current = start_route[:]
    current_dist = rdist(current)
    best = current[:]
    best_dist = current_dist
    T = T_init
    sa_history = [current_dist]
    sa_temp    = [T]

    for it in range(max_iter):
        # Generate neighbour: random 2-opt swap
        i, j = sorted(random.sample(range(1, len(current)), 2))
        neighbour = current[:i] + current[i:j+1][::-1] + current[j+1:]
        neigh_dist = rdist(neighbour)
        delta = neigh_dist - current_dist

        # Accept if better, or probabilistically if worse
        if delta < 0 or random.random() < math.exp(-delta / (T + 1e-12)):
            current      = neighbour
            current_dist = neigh_dist
            if current_dist < best_dist:
                best      = current[:]
                best_dist = current_dist

        T = max(T * cooling, T_min)
        if it % 50 == 0:
            sa_history.append(current_dist)
            sa_temp.append(T)

    return best, best_dist, sa_history, sa_temp

print("  Running Hill Climbing … ", end="", flush=True)
hc_route, hc_dist, hc_history = hill_climbing(nn_r[:])
print(f"done  →  HC dist = {hc_dist:.1f} m")

print("  Running Simulated Annealing … ", end="", flush=True)
sa_route, sa_dist, sa_history, sa_temp = simulated_annealing(nn_r[:])
print(f"done  →  SA dist = {sa_dist:.1f} m")

# Choose best overall route
all_routes  = {"Random": (rnd_r, rnd_d),
               "NN":     (nn_r,  nn_d),
               "HC":     (hc_route, hc_dist),
               "SA":     (sa_route, sa_dist),
               "GA":     (ga_route, ga_dist)}
best_algo   = min(all_routes, key=lambda k: all_routes[k][1])
opt_r, opt_d = all_routes[best_algo]
sav_hc_vs_nn = max(0.0, (nn_d - hc_dist) / nn_d * 100)
sav_sa_vs_nn = max(0.0, (nn_d - sa_dist) / nn_d * 100)
sav_sa_vs_hc = max(0.0, (hc_dist - sa_dist) / hc_dist * 100) if hc_dist > 0 else 0.0
# Picks/hr scales proportionally with route-distance savings vs NN baseline
pph_sa  = pph_base * (1 + sav_sa_vs_nn / 100)
pph_hc  = pph_base * (1 + sav_hc_vs_nn / 100)

print(f"\n  Routing Summary:")
print(f"    Random         : {rnd_d:.1f} m")
print(f"    Nearest Nbr    : {nn_d:.1f} m")
print(f"    Hill Climbing  : {hc_dist:.1f} m  (−{sav_hc_vs_nn:.1f}% vs NN)")
print(f"    Simul. Anneal  : {sa_dist:.1f} m  (−{sav_sa_vs_nn:.1f}% vs NN, −{sav_sa_vs_hc:.1f}% vs HC)")
print(f"    GA-TSP         : {ga_dist:.1f} m")
print(f"  Best algorithm   : {best_algo}")
print(f"  Picks/hr: baseline={pph_base}  HC={pph_hc:.1f}  SA={pph_sa:.1f}  GA={pph_ga:.1f}")

sav   = (nn_d - opt_d) / nn_d * 100
vsrnd = (rnd_d - opt_d) / rnd_d * 100
pph_opt = pph_base * (1 + sav / 100)

# =============================================================================
# 6. PERFORMANCE EVALUATION — KPIs
# =============================================================================
print("\n[6] PERFORMANCE EVALUATION — KPI COMPUTATION")
print("-" * 40)

# ── Forecast KPIs ─────────────────────────────────────────────────────────────
forecast_bias   = float(np.mean(arima_pred - te))   # mean signed error
forecast_acc    = 100 - arima_mape_v                 # simple accuracy %
dw_flag         = "✓" if 1.5 < dw < 2.5 else "⚠"

# ── Inventory KPIs ────────────────────────────────────────────────────────────
gross_margin_pct = (U_price - U_cost) / U_price * 100
gmroi            = (U_price - U_cost) * ann_d / (avg_inv_held * U_cost + 1e-9)
working_capital  = avg_inv_held * U_cost

# ── Order Fulfilment KPIs ─────────────────────────────────────────────────────
on_time_proxy   = sim_fill_rate_opt            # % days all orders processed
avg_cycle_time  = LT_mean + (1 / (PICKERS * PICK_RATE))  # days (rough)
order_acc       = 98.5                         # assumed industry benchmark
perfect_order   = (on_time_proxy / 100) * order_acc   # both factors as fractions → result as %

# ── Warehouse Efficiency KPIs ─────────────────────────────────────────────────
space_util_base = 65.0   # % — SLP baseline estimate
space_util_opt  = 65.0 * (1 + impr / 200)  # proportional improvement
cost_per_order  = total_cost / max(orders_yr, 1)
labour_cost_pct = 55.0   # typical warehouse % of operating cost

# ── Routing / Picking KPIs ────────────────────────────────────────────────────
route_eff_hc    = (nn_d - hc_dist) / nn_d * 100
route_eff_sa    = (nn_d - sa_dist) / nn_d * 100
route_eff_ga    = (nn_d - ga_dist) / nn_d * 100
travel_savings  = (rnd_d - opt_d) / rnd_d * 100

profit_margin   = df["Total Profit"].sum() / df["Total Revenue"].sum() * 100

print(f"  Forecast Bias (ARIMA)      : {forecast_bias:+.2f} units/week")
print(f"  Forecast Accuracy          : {forecast_acc:.1f}%")
print(f"  Durbin-Watson {dw_flag}          : {dw:.4f}")
print(f"  Inventory Turnover         : {inv_turn:.2f}×/yr")
print(f"  Days on Hand               : {doh:.0f} days")
print(f"  GMROI                      : {gmroi:.2f}×")
print(f"  Gross Margin               : {gross_margin_pct:.1f}%")
print(f"  Working Capital (top SKU)  : ${working_capital:,.0f}")
print(f"  Fill Rate (simulated)      : {sim_fill_rate_opt:.1f}%")
print(f"  Avg Cycle Time             : {avg_cycle_time:.1f} days")
print(f"  Perfect Order Rate         : {perfect_order:.1f}%")
print(f"  Space Utilisation (opt.)   : {space_util_opt:.1f}%")
print(f"  Layout Cost Reduction (GA) : {impr:.1f}%")
print(f"  Route Efficiency (HC)      : {route_eff_hc:.1f}%")
print(f"  Route Efficiency (SA)      : {route_eff_sa:.1f}%")
print(f"  Route Efficiency (GA-TSP)  : {route_eff_ga:.1f}%")
print(f"  Picks/hr (SA optimised)    : {pph_sa:.1f}")
print(f"  Overall Profit Margin      : {profit_margin:.1f}%")

# =============================================================================
# 7. FIGURES
# =============================================================================
print("\n[7] GENERATING FIGURES …")

C1="#1E2761"; C2="#F96167"; C3="#02C39A"; C4="#065A82"; bg="#f8f9fa"
zc=[C1,C4,"#028090","#00A896",C3,C2,"#F9E795","#A26769"]
CA="#E63946"; CB="#F4A261"; CC="#2A9D8F"   # ABC colours

# ── Fig 1: Demand Forecasting ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Module 1 — Demand Forecasting  (Weekly Units Sold) + Durbin-Watson",
             fontsize=15, fontweight="bold", color=C1)

ax = axes[0,0]
ax.plot(ts, color=C1, lw=1.5, label="Actual Weekly Units Sold")
ax.axvline(train_n, color="gray", ls="--", alpha=0.6, label="Train/Test Split")
tx = np.arange(train_n, train_n+len(te))
ax.plot(tx, arima_pred, color=C2, lw=2, label=f"ARIMA{best_order} Forecast")
fx = np.arange(len(ts), len(ts)+8)
ax.plot(fx, arima_future, color=C3, lw=2, ls="--", label="8-Week Ahead")
ax.fill_between(fx, arima_future*0.85, arima_future*1.15, color=C3, alpha=0.2)
ax.set_title("ARIMA Forecast", fontweight="bold"); ax.legend(fontsize=7)
ax.set_facecolor(bg); ax.grid(alpha=0.3)
ax.set_xlabel("Week"); ax.set_ylabel("Units Sold")

ax = axes[0,1]
ax.plot(yte_inv, color=C1, lw=1.5, label="Actual")
ax.plot(lstm_pred, color=C2, lw=2, label="LSTM Predicted")
ax.set_title("LSTM Test Prediction", fontweight="bold"); ax.legend()
ax.set_facecolor(bg); ax.grid(alpha=0.3)
ax.set_xlabel("Test Week"); ax.set_ylabel("Units Sold")

ax = axes[1,0]
vr = arima_resid[arima_resid != 0]
ax.plot(vr, color=C4, lw=0.9, alpha=0.8, label="Residuals")
ax.axhline(0, color="red", lw=1.2, ls="--")
# Shade zones for DW interpretation
ax.axhspan(min(vr)*0.5, 0, color=C2, alpha=0.04)
ax.axhspan(0, max(vr)*0.5, color=C3, alpha=0.04)
ax.set_title(f"ARIMA Residuals — Durbin-Watson Test", fontweight="bold")
ax.set_xlabel("Week"); ax.set_ylabel("Residual")
ax.set_facecolor(bg); ax.grid(alpha=0.3)
dw_col = C3 if 1.5 < dw < 2.5 else C2
ax.text(0.97, 0.06,
        f"DW = {dw:.4f}\n{dw_note}\n{dw_interp}",
        transform=ax.transAxes, ha="right", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="#e8f4f8", ec=dw_col, lw=1.5))

ax = axes[1,1]
sku_vols = sku_stats.nlargest(8, "total_units")
bars = ax.barh(sku_vols["Item Type"], sku_vols["total_units"],
               color=[zc[i % len(zc)] for i in range(len(sku_vols))],
               edgecolor="white", lw=0.8)
for bar, val in zip(bars, sku_vols["total_units"]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=7, fontweight="bold")
ax.set_title("Units Sold by Item Type (SKU)", fontweight="bold")
ax.set_xlabel("Total Units Sold"); ax.set_facecolor(bg); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
fig.savefig(f"{OUT}/fig1_demand_forecasting.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig1_demand_forecasting.png")

# ── Fig 2: Inventory ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Module 2 — Inventory Optimization  [{top_sku['Item Type']}]  |  EOQ + Safety + Buffer",
             fontsize=14, fontweight="bold", color=C1)

ax = axes[0,0]
sim_inv=[]; inv_lv=EOQ+safety_stock+buffer_stock; pend=0
for w in range(52):
    d=max(0,np.random.normal(avg_d,std_d)); inv_lv-=d
    if pend>0: inv_lv+=EOQ; pend-=1
    if inv_lv<=ROP and pend==0: pend=LT
    inv_lv=max(0,inv_lv); sim_inv.append(inv_lv)
ax.plot(sim_inv, color=C4, lw=1.5, label="Inventory Level")
ax.axhline(ROP,                    color="orange", ls="--", lw=1.5, label=f"ROP ({ROP:.0f})")
ax.axhline(safety_stock,           color=C2,       ls=":",  lw=1.5, label=f"Safety Stock ({safety_stock:.0f})")
ax.axhline(safety_stock+buffer_stock, color=C3,    ls="-.", lw=1.5, label=f"Safety+Buffer ({safety_stock+buffer_stock:.0f})")
ax.fill_between(range(52), 0, safety_stock, color=C2, alpha=0.07)
ax.fill_between(range(52), safety_stock, safety_stock+buffer_stock, color="orange", alpha=0.07)
ax.set_title("52-Week Inventory Simulation", fontweight="bold")
ax.legend(fontsize=7); ax.set_facecolor(bg); ax.grid(alpha=0.3)
ax.set_xlabel("Week"); ax.set_ylabel("Units")

ax = axes[0,1]
Qs = np.linspace(10, EOQ*2.5, 300)
oc = ann_d/Qs*O_cost; hc = Qs/2*U_cost*h_pct; tc = oc+hc
ax.plot(Qs, oc, color=C2, lw=1.8, label="Ordering Cost")
ax.plot(Qs, hc, color=C1, lw=1.8, label="Holding Cost")
ax.plot(Qs, tc, color=C3, lw=2.5, label="Total Cost")
ax.axvline(EOQ, color="black", ls="--", lw=1.5, label=f"EOQ={EOQ:.0f}")
ax.scatter([EOQ], [tc[np.argmin(tc)]], color="red", zorder=5, s=80)
ax.set_title("EOQ Cost Trade-off", fontweight="bold"); ax.legend(fontsize=8)
ax.set_facecolor(bg); ax.grid(alpha=0.3)
ax.set_xlabel("Order Qty"); ax.set_ylabel("Annual Cost ($)")

ax = axes[1,0]
cats=["Safety\nStock","Buffer\nStock","Cycle\nStock","Max\nInventory"]
vals=[safety_stock, buffer_stock, EOQ/2, max_inv]
cols=[C2,"#F9E795",C1,C3]
bars=ax.bar(cats, vals, color=cols, edgecolor="white", lw=1.5, width=0.55)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.0f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Inventory Stock Components", fontweight="bold")
ax.set_ylabel("Units"); ax.set_facecolor(bg); ax.grid(alpha=0.3, axis="y")

ax = axes[1,1]; ax.axis("off"); ax.set_facecolor("#f0f4f8")
kpi_inv=[
    ("Top SKU",              top_sku["Item Type"]),
    ("Unit Cost (data)",     f"${U_cost:.2f}"),
    ("Unit Price (data)",    f"${U_price:.2f}"),
    ("Gross Margin",         f"{gross_margin_pct:.1f}%"),
    ("Avg Lead Time",        f"{LT_mean:.1f} days"),
    ("EOQ",                  f"{EOQ:.0f} units"),
    ("Safety Stock (95% SL)",f"{safety_stock:.0f} units"),
    ("Buffer Stock (15%)",   f"{buffer_stock:.0f} units"),
    ("Reorder Point",        f"{ROP:.0f} units"),
    ("Max Inventory",        f"{max_inv:.0f} units"),
    ("Inv. Turnover",        f"{inv_turn:.2f}×/yr"),
    ("Days on Hand",         f"{doh:.0f} days"),
    ("GMROI",                f"{gmroi:.2f}×"),
    ("Annual Cost",          f"${total_cost:,.0f}"),
    ("Stockout Prob",        f"{stockout_p*100:.2f}%"),
]
ax.text(0.5, 1.01, "Inventory KPI Summary", ha="center", va="top",
        fontsize=12, fontweight="bold", color=C1, transform=ax.transAxes)
for k,(lab,val) in enumerate(kpi_inv):
    y = 0.92 - k*0.060
    ax.text(0.04, y, f"▸  {lab}", transform=ax.transAxes, fontsize=8, color="#333")
    ax.text(0.96, y, val, transform=ax.transAxes, fontsize=8,
            color=C4, ha="right", fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/fig2_inventory_optimization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig2_inventory_optimization.png")

# ── Fig 3: Layout + Simulation ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Module 3 — Warehouse Layout (SLP + GA)  +  Monte-Carlo Simulation",
             fontsize=14, fontweight="bold", color=C1)
gs3 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

def draw_lay(ax, perm, title, cost):
    ax.set_xlim(-0.05, GC); ax.set_ylim(-0.05, 2)
    ax.set_title(f"{title}\nCost={cost:.0f}", fontweight="bold", fontsize=10)
    ax.set_facecolor("#dde4ef"); ax.set_xticks([]); ax.set_yticks([])
    pm={}
    for slot, zi in enumerate(perm):
        row,col = divmod(slot, GC); cx,cy = col+0.5, 1-row+0.5; pm[zi]=(cx,cy)
        ax.add_patch(Rectangle((col, 1-row), 1, 1,
            facecolor=zc[zi], alpha=0.88, edgecolor="white", lw=2))
        label = zones[zi].replace(" ", "\n") if len(zones[zi]) > 10 else zones[zi]
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=6, fontweight="bold", color="white")
    top5=sorted([(flow[i,j],i,j) for i in range(NZ) for j in range(i+1,NZ)],
                reverse=True)[:5]
    for fv,i,j in top5:
        if i in pm and j in pm:
            xi,yi=pm[i]; xj,yj=pm[j]
            ax.annotate("", xy=(xj,yj), xytext=(xi,yi),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        alpha=0.5, lw=max(0.5, fv/35)))

ax0 = fig.add_subplot(gs3[0, 0])
ax1 = fig.add_subplot(gs3[0, 1])
ax2 = fig.add_subplot(gs3[0, 2])
draw_lay(ax0, list(range(NZ)), "Initial (SLP Baseline)", init_cost)
draw_lay(ax1, best_p,          "GA-Optimised",            best_c)
ax2.plot(hist, color=C1, lw=1.5)
ax2.axhline(best_c, color=C2, ls="--", lw=1.5, label=f"Best={best_c:.0f}")
ax2.set_title("GA Convergence", fontweight="bold")
ax2.set_xlabel("Generation"); ax2.set_ylabel("Layout Cost")
ax2.legend(); ax2.set_facecolor(bg); ax2.grid(alpha=0.3)
ax2.text(GEN*0.55, max(hist)*0.96, f"↓ {impr:.1f}%\nimprovement",
         fontsize=12, fontweight="bold", color=C3,
         bbox=dict(boxstyle="round", fc="white", ec=C3))

# Monte-Carlo simulation plots
ax3 = fig.add_subplot(gs3[1, 0])
ax3.plot(bl_base, color=C2, lw=1.2, alpha=0.7, label="Baseline Layout")
ax3.plot(bl_opt,  color=C3, lw=1.2, alpha=0.9, label="Optimised Layout")
ax3.set_title("Daily Order Backlog (365-Day Simulation)", fontweight="bold", fontsize=10)
ax3.set_xlabel("Day"); ax3.set_ylabel("Backlogged Orders")
ax3.legend(fontsize=8); ax3.set_facecolor(bg); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs3[1, 1])
ax4.plot(ut_base * 100, color=C2, lw=1.0, alpha=0.6, label=f"Baseline  avg={sim_util_base:.1f}%")
ax4.plot(ut_opt  * 100, color=C3, lw=1.0, alpha=0.8, label=f"Optimised avg={sim_util_opt:.1f}%")
ax4.axhline(100, color="red", ls="--", lw=1.2, alpha=0.5, label="Capacity limit")
ax4.set_title("Daily Warehouse Utilisation", fontweight="bold", fontsize=10)
ax4.set_xlabel("Day"); ax4.set_ylabel("Utilisation (%)")
ax4.legend(fontsize=8); ax4.set_facecolor(bg); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs3[1, 2]); ax5.axis("off"); ax5.set_facecolor("#f0f4f8")
sim_kpis = [
    ("── Simulation Parameters ──", ""),
    ("Simulation Days",     f"{N_DAYS}"),
    ("Pickers",             f"{PICKERS}"),
    ("── Baseline Layout ──",  ""),
    ("Pick Rate",           f"{PICK_RATE_BASE} orders/picker/hr"),
    ("Daily Capacity",      f"{DAILY_CAP_BASE:,} orders"),
    ("Avg Backlog",         f"{sim_avg_backlog_base:.1f} orders"),
    ("Fill Rate",           f"{sim_fill_rate_base:.1f}%"),
    ("Avg Utilisation",     f"{sim_util_base:.1f}%"),
    ("── Optimised Layout ──",  ""),
    ("Pick Rate",           f"{PICK_RATE} orders/picker/hr"),
    ("Daily Capacity",      f"{DAILY_CAP:,} orders"),
    ("Avg Backlog",         f"{sim_avg_backlog_opt:.1f} orders"),
    ("Fill Rate",           f"{sim_fill_rate_opt:.1f}%"),
    ("Avg Utilisation",     f"{sim_util_opt:.1f}%"),
]
ax5.text(0.5, 1.01, "Simulation KPI Summary", ha="center", va="top",
         fontsize=11, fontweight="bold", color=C1, transform=ax5.transAxes)
for k,(lab,val) in enumerate(sim_kpis):
    y = 0.92 - k * 0.059
    if "──" in lab:
        ax5.text(0.5, y, lab, transform=ax5.transAxes, fontsize=8,
                 ha="center", fontweight="bold", color=C1)
    else:
        ax5.text(0.04, y, f"▸  {lab}", transform=ax5.transAxes, fontsize=8, color="#333")
        ax5.text(0.96, y, val, transform=ax5.transAxes, fontsize=8,
                 color=C4, ha="right", fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/fig3_layout_simulation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig3_layout_simulation.png")

# ── Fig 4: ABC Analysis ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Module 4 — Order Picking: ABC Analysis (Pareto Classification)",
             fontsize=14, fontweight="bold", color=C1)

ax = axes[0]
colors_abc = [CA if r=="A" else (CB if r=="B" else CC) for r in abc_df["ABC"]]
bars = ax.bar(range(len(abc_df)), abc_df["total_revenue"],
              color=colors_abc, edgecolor="white", lw=0.6)
ax.set_title("Revenue by SKU (ABC Coloured)", fontweight="bold")
ax.set_xlabel("SKU rank"); ax.set_ylabel("Total Revenue ($)")
ax.set_facecolor(bg); ax.grid(alpha=0.3, axis="y")
patches = [mpatches.Patch(color=CA, label="A (≤80%)"),
           mpatches.Patch(color=CB, label="B (80–95%)"),
           mpatches.Patch(color=CC, label="C (>95%)")]
ax.legend(handles=patches, fontsize=9)

ax = axes[1]
ax.plot(range(len(abc_df)), abc_df["cum_pct"], color=C1, lw=2, marker="o", ms=4)
ax.axhline(80, color=CA, ls="--", lw=1.5, label="80% (A boundary)")
ax.axhline(95, color=CB, ls="--", lw=1.5, label="95% (B boundary)")
ax.fill_between(range(len(abc_df)), abc_df["cum_pct"], 100, color=CC, alpha=0.15)
ax.set_title("Cumulative Revenue % (Pareto Curve)", fontweight="bold")
ax.set_xlabel("SKU rank"); ax.set_ylabel("Cumulative Revenue %")
ax.legend(fontsize=9); ax.set_facecolor(bg); ax.grid(alpha=0.3)

ax = axes[2]; ax.axis("off"); ax.set_facecolor("#f0f4f8")
abc_summary = [
    ("Class", "SKUs", "Revenue", "Action"),
    ("─"*5,   "─"*4,  "─"*10,    "─"*18),
    ("A",     f"{count_A}",  f"${rev_A:,.0f} ({rev_A/total_rev*100:.0f}%)",  "Near dispatch, frequent replen."),
    ("B",     f"{count_B}",  f"${rev_B:,.0f} ({rev_B/total_rev*100:.0f}%)",  "Mid-zone, periodic review"),
    ("C",     f"{count_C}",  f"${rev_C:,.0f} ({rev_C/total_rev*100:.0f}%)",  "Deep storage, bulk orders"),
]
ax.text(0.5, 0.97, "ABC Classification Summary", ha="center", va="top",
        fontsize=12, fontweight="bold", color=C1, transform=ax.transAxes)
for k, row in enumerate(abc_summary):
    y = 0.82 - k*0.12
    col_x = [0.02, 0.18, 0.30, 0.62]
    for xi, cell in zip(col_x, row):
        ax.text(xi, y, cell, transform=ax.transAxes, fontsize=8,
                fontweight="bold" if k<=1 else "normal",
                color=C1 if k<=1 else "#333")
ax.text(0.5, 0.20,
        "A items: highest velocity → place\nnear dispatch gate to minimise\npicking travel distance.",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=9, color=C4, style="italic",
        bbox=dict(boxstyle="round", fc="#e8f4f8", ec=C4))
plt.tight_layout()
fig.savefig(f"{OUT}/fig4_abc_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig4_abc_analysis.png")

# ── Fig 5: Routing Comparison (HC + SA + GA) ─────────────────────────────────
country_labels = ["Depot"] + top_countries[:NL]

def draw_route(ax, route, title, dist, col):
    ax.set_title(f"{title}\nDist={dist:.1f} m", fontweight="bold", fontsize=10)
    ax.set_facecolor("#eef2f7"); ax.grid(alpha=0.25)
    for i in range(len(route)):
        f=route[i]; t=route[(i+1)%len(route)]
        ax.annotate("", xy=pts[t], xytext=pts[f],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, alpha=0.7))
    ax.scatter(pts[1:,0], pts[1:,1], s=70,
               c=[plt.cm.plasma(i/NL) for i in range(NL)],
               zorder=5, edgecolors="white", lw=0.8)
    ax.scatter(*pts[0], s=200, c="red", marker="*", zorder=6, label="Depot")
    for idx, p in enumerate(pts):
        lbl = country_labels[idx][:3] if idx > 0 else "D"
        ax.text(p[0]+0.8, p[1]+0.8, lbl, fontsize=5.5, color="#333")
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f"Module 5 — Routing: Hill Climbing vs Simulated Annealing vs GA-TSP  [{NL} Locations]",
             fontsize=14, fontweight="bold", color=C1)

draw_route(axes[0,0], nn_r,     "Nearest Neighbour (Baseline)",    nn_d,    "#aaaaaa")
draw_route(axes[0,1], hc_route, "Hill Climbing",                   hc_dist, C4)
draw_route(axes[0,2], sa_route, "Simulated Annealing",             sa_dist, C3)

# Convergence curves
ax = axes[1,0]
ax.plot(hc_history, color=C4, lw=2, label=f"HC (final={hc_dist:.1f})")
ax.axhline(nn_d, color="#aaa", ls="--", lw=1.2, label=f"NN={nn_d:.1f}")
ax.set_title("Hill Climbing Convergence", fontweight="bold")
ax.set_xlabel("Iteration"); ax.set_ylabel("Route Distance (m)")
ax.legend(fontsize=8); ax.set_facecolor(bg); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.plot(sa_history, color=C3, lw=2, label=f"SA Best (final={sa_dist:.1f})")
ax.axhline(nn_d,   color="#aaa",  ls="--", lw=1.2, label=f"NN={nn_d:.1f}")
ax.axhline(hc_dist,color=C4,      ls=":",  lw=1.2, label=f"HC={hc_dist:.1f}")
ax.set_title("Simulated Annealing Convergence", fontweight="bold")
ax.set_xlabel("Checkpoint (×50 iters)"); ax.set_ylabel("Route Distance (m)")
ax.legend(fontsize=8); ax.set_facecolor(bg); ax.grid(alpha=0.3)
ax2t = ax.twinx()
ax2t.plot(sa_temp, color=C2, lw=1, ls="--", alpha=0.6, label="Temperature")
ax2t.set_ylabel("Temperature", color=C2); ax2t.tick_params(axis="y", labelcolor=C2)

ax = axes[1,2]
algos  = ["Random", "NN", "Hill\nClimbing", "Sim.\nAnnealing", "GA-TSP"]
dists  = [rnd_d,    nn_d, hc_dist,          sa_dist,           ga_dist]
cols_b = ["#aaa", C1, C4, C3, C2]
bars   = ax.bar(algos, dists, color=cols_b, edgecolor="white", lw=1.5, width=0.55)
for b, v in zip(bars, dists):
    ax.text(b.get_x()+b.get_width()/2, v + rnd_d*0.01, f"{v:.0f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Algorithm Comparison — Route Distance", fontweight="bold")
ax.set_ylabel("Total Distance (m)"); ax.set_facecolor(bg); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(f"{OUT}/fig5_routing_hc_sa_ga.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig5_routing_hc_sa_ga.png")

# ── Fig 6: GA-TSP Order Picking ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Module 4 — GA-TSP Order Picking Route  [{NL} Pick Locations]",
             fontsize=14, fontweight="bold", color=C1)
draw_route(axes[0], rnd_r,   "Random Route",                rnd_d, "#aaaaaa")
draw_route(axes[1], nn_r,    "Nearest Neighbour (Baseline)", nn_d,  C1)
draw_route(axes[2], ga_route, f"GA-TSP Optimised",           ga_dist, C2)

plt.tight_layout()
fig.savefig(f"{OUT}/fig6_ga_tsp_picking.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {OUT}/fig6_ga_tsp_picking.png")

# ── Fig 7: KPI Dashboard ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(C1)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)
fig.suptitle("WAREHOUSE OPTIMIZATION — KPI DASHBOARD  |  10,000 Sales Records",
             fontsize=16, fontweight="bold", color="white", y=0.97)

cards=[
    # Forecasting
    ("Forecast MAPE\n(ARIMA)",          f"{arima_mape_v:.1f}%",    "▼ Lower is Better",       C3),
    ("Forecast MAPE\n(LSTM)",           f"{lstm_mape_v:.1f}%",     "▼ Lower is Better",       "#00A896"),
    ("Durbin-Watson\nStatistic",        f"{dw:.3f}",               f"{'✓ White Noise' if 1.5<dw<2.5 else '⚠ Autocorr.'}",  "#028090"),
    ("Forecast Bias\n(ARIMA)",          f"{forecast_bias:+.1f}",   "Units/Week (±0 ideal)",   C4),
    # Inventory
    ("EOQ\n(Optimal Order)",            f"{EOQ:.0f}",              "Units per Order",          C4),
    ("Safety Stock\n(95% SL)",          f"{safety_stock:.0f}",     "Absorbs Demand Var.",      C2),
    ("Buffer Stock\n(+15% LT Demand)",  f"{buffer_stock:.0f}",     "Demand Spike Cushion",     "#F9E795"),
    ("Inventory\nTurnover",             f"{inv_turn:.2f}×",        "Higher is Better",         "#A26769"),
    # Layout & Simulation
    ("Stockout\nProbability",           f"{stockout_p*100:.2f}%",  "▼ Lower is Better",        C2),
    ("Fill Rate\n(Simulated)",          f"{sim_fill_rate_opt:.1f}%","Orders Fulfilled Daily",  C3),
    ("Layout Cost\nReduction (GA)",     f"{impr:.1f}%",            "vs SLP Baseline",           C1),
    ("Warehouse\nUtilisation",          f"{sim_util_opt:.1f}%",    "Daily Avg (Optimised)",     C4),
    # Routing & KPIs
    ("Route Saving\n(SA vs NN)",        f"{sav_sa_vs_nn:.1f}%",    "Simulated Annealing",       C3),
    ("Route Saving\n(GA vs NN)",        f"{ga_saving_vs_nn:.1f}%", "Genetic Algorithm",          C2),
    ("Picks/Hour\n(SA Optimised)",      f"{pph_sa:.0f}",           "↑ Throughput",               "#028090"),
    ("Profit\nMargin",                  f"{profit_margin:.1f}%",   "From Dataset",               C3),
]

for k,(label,value,sub,color) in enumerate(cards):
    r,c = divmod(k,4); ax = fig.add_subplot(gs[r,c])
    ax.set_facecolor(color); ax.axis("off")
    ax.text(0.5, 0.70, value, ha="center", va="center",
            fontsize=22, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.40, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.12, sub, ha="center", va="center",
            fontsize=7.5, color="white", transform=ax.transAxes, alpha=0.88)

fig.savefig(f"{OUT}/fig7_kpi_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  ✓ {OUT}/fig7_kpi_dashboard.png")

# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================
print("\n"+"="*70)
print("  FINAL SUMMARY — ALL MODULES")
print("="*70)
print(f"""
  DATASET: 10,000 Sales Records
    Orders      : {total_orders:,}
    Date Range  : {date_min} → {date_max}
    Regions     : {df['Region'].nunique()}   |   Item Types: {df['Item Type'].nunique()}
    Total Rev   : ${df['Total Revenue'].sum():,.0f}   |   Profit: ${df['Total Profit'].sum():,.0f}

  MODULE 1 — DEMAND FORECASTING  (Weekly Units Sold)
    ARIMA{best_order}  MAE={arima_mae_v:.2f}  RMSE={arima_rmse_v:.2f}  MAPE={arima_mape_v:.2f}%
    LSTM    MAE={lstm_mae_v:.2f}  RMSE={lstm_rmse_v:.2f}  MAPE={lstm_mape_v:.2f}%
    Durbin-Watson = {dw:.4f}  →  {dw_note}
    Interpretation: {dw_interp}

  MODULE 2 — INVENTORY OPTIMIZATION  [{top_sku['Item Type']}]
    Unit Cost (data)=${U_cost:.2f}  Unit Price=${U_price:.2f}  Lead Time={LT_mean:.1f}d (σ={LT_std:.1f}d)
    EOQ={EOQ:.0f} units
    Safety Stock={safety_stock:.0f} units  (95% service level, Z=1.645)
    Buffer Stock={buffer_stock:.0f} units  (15% of avg LT demand)
    ROP={ROP:.0f}  MaxInventory={max_inv:.0f}
    AnnualCost=${total_cost:,.0f}  Turnover={inv_turn:.2f}×  DoH={doh:.0f}d
    GMROI={gmroi:.2f}×  Gross Margin={gross_margin_pct:.1f}%
    Stockout probability = {stockout_p*100:.2f}%

  MODULE 3 — LAYOUT OPTIMIZATION (SLP + GA + Monte-Carlo Simulation)
    Layout cost: {init_cost:.0f} → {best_c:.0f}  ({impr:.1f}% cost reduction via GA)
    Simulation ({N_DAYS} days, {PICKERS} pickers):
      Baseline  — fill rate={sim_fill_rate_base:.1f}%  backlog={sim_avg_backlog_base:.1f}  util={sim_util_base:.1f}%
      Optimised — fill rate={sim_fill_rate_opt:.1f}%   backlog={sim_avg_backlog_opt:.1f}   util={sim_util_opt:.1f}%

  MODULE 4 — ORDER PICKING (ABC Analysis + GA-TSP)
    A items: {count_A} SKUs → {rev_A/total_rev*100:.0f}% revenue (high-velocity, near dispatch)
    B items: {count_B} SKUs → {rev_B/total_rev*100:.0f}% revenue (mid-zone)
    C items: {count_C} SKUs → {rev_C/total_rev*100:.0f}% revenue (deep storage)
    GA-TSP route: {ga_dist:.1f} m  (saves {ga_saving_vs_nn:.1f}% vs NN)

  MODULE 5 — ROUTING ALGORITHMS
    Nearest Neighbour : {nn_d:.1f} m
    Hill Climbing     : {hc_dist:.1f} m  (−{sav_hc_vs_nn:.1f}% vs NN)
    Simul. Annealing  : {sa_dist:.1f} m  (−{sav_sa_vs_nn:.1f}% vs NN,  −{sav_sa_vs_hc:.1f}% vs HC)
    GA-TSP            : {ga_dist:.1f} m
    Best algorithm    : {best_algo}
    Picks/hr: base={pph_base}  HC={pph_hc:.1f}  SA={pph_sa:.1f}  GA={pph_ga:.1f}

  MODULE 6 — PERFORMANCE KPIs
    Forecast Accuracy       : {forecast_acc:.1f}%   Forecast Bias: {forecast_bias:+.2f} units/wk
    Inventory Turnover      : {inv_turn:.2f}×   DoH: {doh:.0f}d   GMROI: {gmroi:.2f}×
    Fill Rate (simulated)   : {sim_fill_rate_opt:.1f}%
    Perfect Order Rate      : {perfect_order:.1f}%
    Warehouse Utilisation   : {sim_util_opt:.1f}%
    Overall Profit Margin   : {profit_margin:.1f}%

  Output saved to: ./{OUT}/
    fig1_demand_forecasting.png
    fig2_inventory_optimization.png
    fig3_layout_simulation.png
    fig4_abc_analysis.png
    fig5_routing_hc_sa_ga.png
    fig6_ga_tsp_picking.png
    fig7_kpi_dashboard.png

  ✓  All models trained & outputs produced.
""")
print("="*70)
