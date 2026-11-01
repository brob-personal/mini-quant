"""
portfolio_engine.py

Daily event-driven simulator for the levers in portfolio_levers.py.

Modeled
  * Tax-lot accounting per account (FIFO/LIFO/HIFO), ST/LT gains, collectibles (physical
    bullion ETFs), 60/40 for index options, loss carryforward pool, dividend/interest tax
    drag (taxed on receipt), Roth/Trad terminal tax, IRA cap with overflow to a taxable account.
  * Contributions (lump / monthly / schedule), cash-flow-first rebalancing, calendar and band
    rebalancing, slippage/commission, DRIP or idle cash.
  * LEAP ladder: Black-Scholes marks, strike chosen by target delta, staggered rungs, rolls at
    roll_at_dte, option spread/commission, 1256 or ordinary holding-period treatment.

Simplifications (all conservative or small; know them before trusting a number)
  * Taxes are paid immediately at realization / dividend receipt, flat marginal rates, no bracket
    stacking. Loss pool offsets later gains only (no same-year refund, no $3k ordinary deduction).
  * Wash sales and Section 1256 year-end mark-to-market are NOT modeled.
  * Idle cash (DRIP off) earns nothing. Option strikes are not rounded to listed increments;
    fractional contracts allowed. Option IV is flat or realized-vol x multiplier, not a surface.
  * Trades execute at the day's close.
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np
import pandas as pd

from portfolio_levers import (COLLECTIBLE_LEAVES, TICKERS, AccountType, LotMethod, Rebalance,
                              SimConfig)

_N = NormalDist()


# ----------------------------------------------------------------------------
# Option math
# ----------------------------------------------------------------------------

def bs_price(S, K, T, r, q, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    if kind == "call":
        return S * math.exp(-q * T) * _N.cdf(d1) - K * math.exp(-r * T) * _N.cdf(d2)
    return K * math.exp(-r * T) * _N.cdf(-d2) - S * math.exp(-q * T) * _N.cdf(-d1)


def strike_for_delta(S, T, r, q, sigma, delta, kind="call"):
    """Strike whose |delta| equals `delta` under Black-Scholes."""
    p = min(delta * math.exp(q * T), 0.999999)
    d1 = _N.inv_cdf(p) if kind == "call" else -_N.inv_cdf(p)
    return S * math.exp(-(d1 * sigma * math.sqrt(T)) + (r - q + 0.5 * sigma * sigma) * T)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

@dataclass
class MarketData:
    """prices: unadjusted (split-adjusted) closes. dividends: cash per share on ex-date.
    Columns are leaf keys (e.g. 'CORE_SP500') plus the LEAP underlying and benchmark tickers.
    rf: annual T-bill yield as decimal (optional). cpi: index level (optional, for real values)."""
    prices: pd.DataFrame
    dividends: pd.DataFrame
    rf: pd.Series | None = None
    cpi: pd.Series | None = None


def required_columns(cfg, targets=None):
    targets = targets or cfg.portfolio.resolve()
    cols = {}
    for leaf in targets:
        if leaf == "LEAP_LADDER":
            u = cfg.portfolio.leap.underlying
            cols[u] = u
        else:
            cols[leaf] = TICKERS.get(leaf, leaf)
    cols[cfg.time.benchmark] = cfg.time.benchmark
    return cols


def load_yahoo(cfg, lookback_days=400, fetch_rf=True):
    """Pull prices/dividends via yfinance (pip install yfinance). NOT exercised in the dev
    sandbox (no Yahoo access); treat as untested. Lookback gives realized-vol history for LEAPs."""
    import yfinance as yf
    start = pd.Timestamp(cfg.time.start) - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(cfg.time.end) + pd.Timedelta(days=1)
    cache, px, dv = {}, {}, {}
    for col, tkr in required_columns(cfg).items():
        if tkr not in cache:
            h = yf.Ticker(tkr).history(start=start, end=end, auto_adjust=False, actions=True)
            if h.empty:
                raise ValueError(f"no data returned for {tkr}")
            h.index = h.index.tz_localize(None).normalize()
            cache[tkr] = h
        px[col], dv[col] = cache[tkr]["Close"], cache[tkr]["Dividends"]
    prices = pd.DataFrame(px).sort_index()
    divs = pd.DataFrame(dv).reindex(prices.index).fillna(0.0)
    rf = None
    if fetch_rf:
        try:
            h = yf.Ticker("^IRX").history(start=start, end=end)
            h.index = h.index.tz_localize(None).normalize()
            rf = h["Close"] / 100.0
        except Exception:
            rf = None
    return MarketData(prices, divs, rf)


def synthetic_data(cfg, seed=0, lookback_days=400):
    """Correlated GBM stand-in for testing the plumbing. Results carry no real-world meaning."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(pd.Timestamp(cfg.time.start) - pd.Timedelta(days=lookback_days),
                         pd.Timestamp(cfg.time.end))
    n = len(idx)
    mkt = rng.normal(0.0, 0.15 / math.sqrt(252), n)
    month_end = np.r_[idx[1:].month != idx[:-1].month, True]
    qtr_end = month_end & np.isin(idx.month, [3, 6, 9, 12])
    prices, divs = {}, {}
    for col in required_columns(cfg):
        if col == "TBILL" or col.startswith("BOND_"):
            mu, vol, beta, dy, freq = 0.03, 0.05, 0.0, 0.03, month_end
            if col == "TBILL":
                mu, vol, dy = 0.02, 0.003, 0.02
        elif col in COLLECTIBLE_LEAVES or "MINERS" in col:
            mu, vol, beta, dy, freq = 0.05, 0.18, 0.3, 0.0, qtr_end
        else:
            mu, vol, beta, dy, freq = 0.09, 0.16, 1.0, 0.015, qtr_end
        idio = max(vol ** 2 - (beta * 0.15) ** 2, 1e-6) ** 0.5
        lr = (mu - dy) / 252 + beta * mkt + rng.normal(0, idio / math.sqrt(252), n) - 0.5 * vol ** 2 / 252
        p = 100 * np.exp(np.cumsum(lr))
        prices[col] = p
        d = np.where(freq, p * dy / (12 if freq is month_end else 4), 0.0)
        divs[col] = d
    return MarketData(pd.DataFrame(prices, index=idx), pd.DataFrame(divs, index=idx))


# ----------------------------------------------------------------------------
# Environment (arrays aligned to the trading calendar)
# ----------------------------------------------------------------------------

class _Env:
    def __init__(self, cfg, data, targets, truncate_start):
        self.cfg = cfg
        prices = data.prices.sort_index().ffill()
        keys = {k for k in targets if k != "LEAP_LADDER"}
        self.leap = cfg.portfolio.leap if "LEAP_LADDER" in targets else None
        if self.leap:
            keys.add(self.leap.underlying)
        missing = [k for k in keys if k not in prices.columns]
        if missing:
            raise KeyError(f"data missing columns: {missing}")
        self.dates = prices.index
        i0 = int(self.dates.searchsorted(pd.Timestamp(cfg.time.start)))
        self.i1 = int(self.dates.searchsorted(pd.Timestamp(cfg.time.end), side="right"))
        self.P = {k: prices[k].to_numpy(float) for k in keys}
        late = {k: int(np.argmax(~np.isnan(self.P[k]))) for k in keys if k != getattr(self.leap, "underlying", None)}
        bad = {k: v for k, v in late.items() if v > i0 or np.isnan(self.P[k][i0])}
        if bad:
            if not truncate_start:
                info = ", ".join(f"{k} ({TICKERS.get(k, k)}) starts {self.dates[v].date()}" for k, v in bad.items())
                raise ValueError(f"data not available at sim start {cfg.time.start}: {info}. "
                                 f"Use truncate_start=True or supply proxy history.")
            i0 = max(bad.values())
        if self.leap and np.isnan(self.P[self.leap.underlying][i0]):
            raise ValueError("LEAP underlying has no price at sim start")
        self.i0 = i0
        divs = data.dividends.reindex(self.dates).fillna(0.0)
        self.D = {k: (divs[k].to_numpy(float) if k in divs.columns else np.zeros(len(self.dates)))
                  for k in keys}
        if self.leap:
            S = self.P[self.leap.underlying]
            if self.leap.iv_mode == "flat":
                iv = np.full(len(S), self.leap.flat_iv)
            else:
                lr = pd.Series(np.log(S)).diff()
                v = lr.rolling(60, min_periods=20).std() * math.sqrt(252) * self.leap.iv_multiplier
                iv = v.fillna(self.leap.flat_iv).clip(lower=0.08).to_numpy()
            self.S, self.iv = S, iv
            if data.rf is not None:
                self.r = data.rf.reindex(self.dates, method="ffill").fillna(self.leap.risk_free).to_numpy(float)
            else:
                self.r = np.full(len(S), self.leap.risk_free)
        self.rf_mean = (float(data.rf.reindex(self.dates, method="ffill").iloc[i0:self.i1].mean())
                        if data.rf is not None else 0.0)
        mend = np.r_[self.dates[1:].month != self.dates[:-1].month, True]
        self.month_end = mend
        self.cpi = (data.cpi.reindex(self.dates, method="ffill").to_numpy(float)
                    if data.cpi is not None else None)


# ----------------------------------------------------------------------------
# Account
# ----------------------------------------------------------------------------

@dataclass
class Lot:
    qty: float
    basis: float      # per unit, includes buy costs
    date: pd.Timestamp


@dataclass
class Contract:
    strike: float
    expiry: pd.Timestamp


def _is_interest(key):
    return key == "TBILL" or key.startswith("BOND_")


class Account:
    def __init__(self, name, env, tax, targets, trades):
        self.name, self.env, self.tax, self.targets = name, env, tax, targets
        self.cfg = env.cfg
        self.leap = env.leap
        self.cash = 0.0
        self.lots = defaultdict(list)
        self.qty = defaultdict(float)
        self.contracts = {}
        self.loss_pool = 0.0
        self.taxes = defaultdict(float)       # gains / dividends / interest (computed)
        self.costs = 0.0
        self.sold_value = 0.0
        self.deposited = 0.0                  # net deposits (basis for Roth early-withdrawal calc)
        self.trades = trades
        self.is1256 = bool(self.leap and self.leap.index_option_1256)

    # --- pricing -----------------------------------------------------------
    def px(self, key, i):
        if key.startswith("LEAP_LADDER#"):
            c = self.contracts.get(int(key.split("#")[1]))
            if c is None:
                return 0.0
            T = max((c.expiry - self.env.dates[i]).days, 0) / 365.0
            L = self.leap
            return 100.0 * bs_price(self.env.S[i], c.strike, T, self.env.r[i], L.dividend_yield,
                                    self.env.iv[i], L.option_type)
        return self.env.P[key][i]

    def _is_opt(self, key):
        return key.startswith("LEAP_LADDER#")

    def _c(self, key):
        c = self.cfg.costs
        return c.option_half_spread_pct if self._is_opt(key) else c.slippage_bps / 1e4

    def _unit_fee(self, key):
        return self.cfg.costs.option_commission_per_contract if self._is_opt(key) else 0.0

    def _fixed(self, key):
        return 0.0 if self._is_opt(key) else self.cfg.costs.commission_per_trade

    def leaf_value(self, leaf, i):
        if leaf == "LEAP_LADDER":
            return sum(self.qty[k] * self.px(k, i) for k in self._rung_keys() if self.qty[k] > 0)
        return self.qty[leaf] * self.px(leaf, i) if self.qty[leaf] > 0 else 0.0

    def _rung_keys(self):
        return [f"LEAP_LADDER#{k}" for k in range(self.leap.rungs)] if self.leap else []

    def value(self, i):
        v = self.cash
        for key, q in self.qty.items():
            if q > 1e-12:
                v += q * self.px(key, i)
        return v

    # --- taxes -------------------------------------------------------------
    def _gain_tax(self, key, gain, hold_days):
        if not self.tax.is_taxable_account:
            return 0.0
        if gain <= 0:
            self.loss_pool += -gain
            return 0.0
        use = min(self.loss_pool, gain)
        self.loss_pool -= use
        rate = self.tax.rate_gain(hold_days, collectible=key in COLLECTIBLE_LEAVES,
                                  section_1256=self._is_opt(key) and self.is1256)
        return (gain - use) * rate

    # --- trading -----------------------------------------------------------
    def buy(self, key, amount, i, d, free=False):
        amount = min(amount, self.cash)
        px = self.px(key, i)
        if amount <= 1e-9 or px <= 0:
            return
        c, fee, fixed = (0.0, 0.0, 0.0) if free else (self._c(key), self._unit_fee(key), self._fixed(key))
        qty = (amount - fixed) / (px * (1 + c) + fee)
        if qty <= 0:
            return
        self.cash -= amount
        self.lots[key].append(Lot(qty, amount / qty, d))
        self.qty[key] += qty
        self.costs += amount - qty * px
        self.trades.append((d, self.name, key, "buy", qty, px, amount, 0.0))

    def _ordered(self, key):
        m = self.tax.lot_method
        lots = self.lots[key]
        if m is LotMethod.FIFO:
            return sorted(lots, key=lambda l: l.date)
        if m is LotMethod.LIFO:
            return sorted(lots, key=lambda l: l.date, reverse=True)
        return sorted(lots, key=lambda l: -l.basis)

    def sell(self, key, qty, i, d):
        qty = min(qty, self.qty[key])
        if qty <= 1e-12:
            return 0.0
        px = self.px(key, i)
        c, fee = self._c(key), self._unit_fee(key)
        net_unit = px * (1 - c) - fee
        rem, proceeds, tax = qty, 0.0, 0.0
        for lot in self._ordered(key):
            if rem <= 1e-12:
                break
            take = min(lot.qty, rem)
            tax += self._gain_tax(key, take * (net_unit - lot.basis), (d - lot.date).days)
            proceeds += take * net_unit
            lot.qty -= take
            rem -= take
        sold = qty - rem
        self.lots[key] = [l for l in self.lots[key] if l.qty > 1e-12]
        self.qty[key] = max(self.qty[key] - sold, 0.0)
        fixed = self._fixed(key)
        proceeds -= fixed
        self.costs += sold * (px - net_unit) + fixed
        self.sold_value += sold * px
        self.taxes["gains"] += tax
        net = proceeds - (tax if self.cfg.tax.tax_paid_from_portfolio else 0.0)
        self.cash += net
        self.trades.append((d, self.name, key, "sell", sold, px, proceeds, tax))
        return net

    def _sell_leaf(self, leaf, value, i, d):
        if leaf == "LEAP_LADDER":
            lv = self.leaf_value(leaf, i)
            if lv <= 0:
                return
            f = min(value / lv, 1.0)
            for k in self._rung_keys():
                self.sell(k, self.qty[k] * f, i, d)
        else:
            self.sell(leaf, value / self.px(leaf, i), i, d)

    def _open_contract(self, k, dte, i):
        L, e = self.leap, self.env
        T = dte / 365.0
        K = strike_for_delta(e.S[i], T, e.r[i], L.dividend_yield, e.iv[i], L.target_delta, L.option_type)
        self.contracts[k] = Contract(K, e.dates[i] + pd.Timedelta(days=dte))

    def _buy_leaf(self, leaf, amount, i, d):
        if leaf == "LEAP_LADDER":
            L = self.leap
            for k, dte in enumerate(L.initial_dtes()):
                if k not in self.contracts:
                    self._open_contract(k, dte, i)
            for k in self._rung_keys():
                self.buy(k, amount / L.rungs, i, d)
        else:
            self.buy(leaf, amount, i, d)

    def roll(self, i, d):
        L = self.leap
        if not L:
            return
        for k, c in list(self.contracts.items()):
            if (c.expiry - d).days <= L.roll_at_dte:
                key = f"LEAP_LADDER#{k}"
                before = self.cash
                self.sell(key, self.qty[key], i, d)
                got = self.cash - before
                self._open_contract(k, L.initial_dte, i)
                self.buy(key, got, i, d)

    def dividends(self, i, d):
        for key in [k for k, q in self.qty.items() if q > 1e-12 and not self._is_opt(k)]:
            dv = self.env.D[key][i]
            if dv <= 0:
                continue
            gross = self.qty[key] * dv
            if _is_interest(key):
                tax, cat = gross * self.tax.rate_interest(key), "interest"
            else:
                tax, cat = gross * self.tax.rate_dividend(True), "dividends"
            self.taxes[cat] += tax
            net = gross - (tax if self.cfg.tax.tax_paid_from_portfolio else 0.0)
            self.cash += net
            if self.cfg.time.reinvest_dividends:
                self.buy(key, net, i, d, free=True)

    def deploy(self, amount, i, d, deficit_first):
        amount = min(amount, self.cash)
        if amount <= 1e-6:
            return
        total = self.value(i)
        cur = {l: self.leaf_value(l, i) for l in self.targets}
        deficit = {l: max(w * total - cur[l], 0.0) for l, w in self.targets.items()}
        sd = sum(deficit.values())
        if deficit_first and sd > 0:
            if sd >= amount:
                alloc = {l: amount * v / sd for l, v in deficit.items()}
            else:
                alloc = {l: deficit[l] + (amount - sd) * self.targets[l] for l in self.targets}
        else:
            alloc = {l: amount * w for l, w in self.targets.items()}
        for l, a in alloc.items():
            self._buy_leaf(l, a, i, d)

    def drift(self, i):
        total = self.value(i)
        if total <= 0:
            return 0.0
        return max(abs(self.leaf_value(l, i) / total - w) for l, w in self.targets.items())

    def rebalance(self, i, d):
        total = self.value(i)
        tol = max(1.0, 1e-4 * total)
        for l, w in self.targets.items():
            ex = self.leaf_value(l, i) - w * total
            if ex > tol:
                self._sell_leaf(l, ex, i, d)
        self.deploy(self.cash, i, d, True)

    def withdraw(self, amount, i, d):
        total = self.value(i)
        if total <= 0:
            return 0.0
        f = min(amount / (total - self.cash) if total > self.cash else 0.0, 1.0)
        for key in [k for k, q in self.qty.items() if q > 1e-12]:
            self.sell(key, self.qty[key] * f, i, d)
        paid = min(amount, self.cash)
        self.cash -= paid
        self.deposited -= paid
        return paid

    def liquidation_tax(self, i, d):
        """Tax if everything were sold and withdrawn at day i (not executed)."""
        t = self.tax
        if t.account_type is not AccountType.TAXABLE:
            return t.terminal_tax(self.value(i), contributions_basis=max(self.deposited, 0.0))
        pool, gains = self.loss_pool, []
        for key, lots in self.lots.items():
            net_unit = self.px(key, i) * (1 - self._c(key)) - self._unit_fee(key)
            for lot in lots:
                g = lot.qty * (net_unit - lot.basis)
                if g <= 0:
                    pool += -g
                else:
                    gains.append((key, g, (d - lot.date).days))
        gains.sort(key=lambda x: x[2])      # short-term first soaks up losses
        total = 0.0
        for key, g, hold in gains:
            use = min(pool, g)
            pool -= use
            total += (g - use) * t.rate_gain(hold, collectible=key in COLLECTIBLE_LEAVES,
                                             section_1256=self._is_opt(key) and self.is1256)
        return total


# ----------------------------------------------------------------------------
# Results / metrics
# ----------------------------------------------------------------------------

def xirr(cfs):
    if len(cfs) < 2:
        return float("nan")
    t0 = cfs[0][0]
    ts = np.array([(d - t0).days / 365.25 for d, _ in cfs])
    amts = np.array([a for _, a in cfs])
    f = lambda r: float(np.sum(amts / (1 + r) ** ts))
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass
class Result:
    label: str
    cfg: SimConfig
    curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

    def summary(self):
        m = self.metrics
        print(f"== {self.label} | {self.cfg.tax.account_type.value} | "
              f"{self.curve.index[0].date()} -> {self.curve.index[-1].date()} ==")
        rows = [("Outlay (after-tax $ in)", m["outlay"], "${:,.0f}"),
                ("Final value (pre-liquidation)", m["final_value"], "${:,.0f}"),
                ("Liquidation tax", m["terminal_tax"], "${:,.0f}"),
                ("Final value after tax", m["after_tax_final"], "${:,.0f}"),
                ("After-tax IRR (money-weighted)", m["irr_after_tax"], "{:.2%}"),
                ("CAGR (time-weighted, pre-liq.)", m["cagr_twr"], "{:.2%}"),
                ("Volatility (ann.)", m["vol"], "{:.2%}"),
                ("Sharpe", m["sharpe"], "{:.2f}"),
                ("Max drawdown", m["max_drawdown"], "{:.2%}"),
                ("Taxes paid: gains / div / interest",
                 None, f"${m['tax_gains']:,.0f} / ${m['tax_dividends']:,.0f} / ${m['tax_interest']:,.0f}"),
                ("Trading costs", m["costs"], "${:,.0f}"),
                ("Annual turnover", m["turnover"], "{:.1%}")]
        for name, v, fmt in rows:
            print(f"  {name:<36}{fmt.format(v) if v is not None else fmt}")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def _contribution_buckets(cfg, env):
    """Map flows to trading-day indices. Returns {i: [main_deposit, overflow_deposit, outlay]}."""
    tax = cfg.tax
    start = env.dates[env.i0].date()
    end = cfg.time.end
    flows = cfg.contributions.cash_flows(start, end)
    eq = tax.account_type is AccountType.TRAD_IRA and tax.trad_equalize_outlay
    t = tax.ordinary_rate + tax.state_rate
    gross = [(d, a / (1 - t) if (eq and a > 0) else a) for d, a in flows]
    main, over = tax.split_contributions(gross)
    k = (1 - t) if eq else 1.0
    out = defaultdict(lambda: [0.0, 0.0, 0.0])
    for lst, col in ((main, 0), (over, 1)):
        for d, a in lst:
            i = int(env.dates.searchsorted(pd.Timestamp(d)))
            i = max(i, env.i0)
            if i >= env.i1:
                continue
            out[i][col] += a
            out[i][2] += a * k
    return out


def run(cfg, data, targets=None, truncate_start=False, label="portfolio"):
    cfg.validate()
    targets = dict(targets or cfg.portfolio.resolve())
    env = _Env(cfg, data, targets, truncate_start)
    buckets = _contribution_buckets(cfg, env)
    trades = []
    main = Account("main", env, cfg.tax, targets, trades)
    overflow = None
    if cfg.tax.account_type is not AccountType.TAXABLE and any(b[1] > 0 for b in buckets.values()):
        overflow = Account("overflow", env, replace(cfg.tax, account_type=AccountType.TAXABLE), targets, trades)
    accts = [main] + ([overflow] if overflow else [])

    n = env.i1 - env.i0
    val, dep, outl = np.zeros(n), np.zeros(n), np.zeros(n)
    freq = cfg.time.rebalance
    band = cfg.time.rebalance_band
    cf_first = cfg.time.rebalance_with_cashflows_first
    for j, i in enumerate(range(env.i0, env.i1)):
        d = env.dates[i]
        for a in accts:
            a.roll(i, d)
            a.dividends(i, d)
        if i in buckets:
            md, od, ou = buckets[i]
            outl[j] = ou
            for a, amt in ((main, md), (overflow, od)):
                if a is None or amt == 0:
                    continue
                if amt > 0:
                    a.cash += amt
                    a.deposited += amt
                    dep[j] += amt
                    a.deploy(a.cash, i, d, cf_first)
                else:
                    dep[j] -= a.withdraw(-amt, i, d)
        if i < env.i1 - 1 and env.month_end[i]:
            m = d.month
            sched = ((freq is Rebalance.MONTHLY) or
                     (freq is Rebalance.QUARTERLY and m in (3, 6, 9, 12)) or
                     (freq is Rebalance.ANNUAL and m == 12))
            for a in accts:
                if a.value(i) > 0 and (sched or (band is not None and a.drift(i) > band)):
                    a.rebalance(i, d)
        val[j] = sum(a.value(i) for a in accts)

    dates = env.dates[env.i0:env.i1]
    r = np.zeros(n)
    for j in range(1, n):
        if val[j - 1] > 1e-9:
            r[j] = (val[j] - dep[j]) / val[j - 1] - 1.0
    idx = np.cumprod(1 + r)
    dd = idx / np.maximum.accumulate(idx) - 1
    live = np.r_[False, val[:-1] > 1e-9]
    first = int(np.argmax(val > 1e-9)) if (val > 1e-9).any() else 0
    years = max((dates[-1] - dates[first]).days / 365.25, 1e-9)
    vol = float(np.std(r[live]) * math.sqrt(252)) if live.sum() > 2 else float("nan")
    excess = (np.mean(r[live]) * 252 - env.rf_mean) if live.any() else float("nan")
    last_i = env.i1 - 1
    term = sum(a.liquidation_tax(last_i, dates[-1]) for a in accts)
    final = float(val[-1])
    cfs = [(dates[j], -outl[j]) for j in range(n) if outl[j] > 0]
    cfs.append((dates[-1], final - term))

    curve = pd.DataFrame({"value": val, "deposits": np.cumsum(dep), "outlay": np.cumsum(outl),
                          "twr_index": idx, "drawdown": dd}, index=dates)
    if cfg.time.inflation_adjust:
        if env.cpi is None:
            raise ValueError("inflation_adjust=True requires MarketData.cpi")
        curve["real_value"] = val / (env.cpi[env.i0:env.i1] / env.cpi[env.i0])
    tr = pd.DataFrame(trades, columns=["date", "account", "key", "side", "qty", "price", "value", "tax"])
    sold = sum(a.sold_value for a in accts)
    metrics = {
        "outlay": float(outl.sum()), "final_value": final, "terminal_tax": float(term),
        "after_tax_final": final - float(term), "irr_after_tax": xirr(cfs),
        "cagr_twr": float(idx[-1] ** (1 / years) - 1) if idx[-1] > 0 else float("nan"),
        "vol": vol, "sharpe": float(excess / vol) if vol and vol == vol and vol > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "tax_gains": sum(a.taxes["gains"] for a in accts),
        "tax_dividends": sum(a.taxes["dividends"] for a in accts),
        "tax_interest": sum(a.taxes["interest"] for a in accts),
        "costs": sum(a.costs for a in accts),
        "turnover": float(sold / max(np.mean(val[val > 0]) if (val > 0).any() else 1.0, 1.0) / years),
        "overflow_deposits": float(sum(b[1] for b in buckets.values())),
    }
    return Result(label, cfg, curve, tr, metrics)


def backtest(cfg, data, truncate_start=False):
    """Portfolio plus a single-asset benchmark under identical tax/contribution rules."""
    p = run(cfg, data, truncate_start=truncate_start, label="portfolio")
    b = run(cfg, data, targets={cfg.time.benchmark: 1.0}, truncate_start=truncate_start,
            label=f"benchmark ({cfg.time.benchmark})")
    return {"portfolio": p, "benchmark": b}


_COLS = ["outlay", "final_value", "terminal_tax", "after_tax_final", "irr_after_tax",
         "cagr_twr", "max_drawdown", "tax_gains", "tax_dividends", "tax_interest"]


def compare_account_types(cfg, data, targets=None, truncate_start=False):
    """Same portfolio and contributions across Taxable / Roth / Trad. Set
    tax.trad_equalize_outlay=True for a like-for-like Roth vs Trad comparison."""
    rows = {}
    for at in AccountType:
        c = replace(cfg, tax=replace(cfg.tax, account_type=at))
        rows[at.value] = run(c, data, targets, truncate_start).metrics
    return pd.DataFrame(rows).T[_COLS]


if __name__ == "__main__":
    from datetime import date

    from portfolio_levers import (ContribMode, ContributionPlan, Detail, PortfolioConfig,
                                  StyleTilt, TaxConfig, TimeConfig)

    cfg = SimConfig(
        time=TimeConfig(start=date(2012, 1, 1), end=date(2024, 12, 31)),
        tax=TaxConfig(account_type=AccountType.TAXABLE),
        contributions=ContributionPlan(mode=ContribMode.MONTHLY, amount=500),
        portfolio=PortfolioConfig(detail=Detail.REGIONS, stocks=0.70, bonds=0.10, metals=0.05, leaps=0.15,
                                  style=StyleTilt(enabled=True, value_tilt=0.5)),
    )
    data = synthetic_data(cfg)          # swap for load_yahoo(cfg) on a machine with network access
    for w in cfg.warnings():
        print("!", w)
    res = backtest(cfg, data)
    res["portfolio"].summary()
    res["benchmark"].summary()
    print(compare_account_types(cfg, data).round(3).to_string())
