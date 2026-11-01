"""
portfolio_levers.py

Configuration layer ("levers") for a personal portfolio simulator. No engine here:
this file defines inputs, validates them, resolves the allocation tree to leaf weights,
and exposes tax-rate helpers and a contribution cash-flow generator. A simulation loop
(custom pandas, vectorbt, etc.) consumes SimConfig.

Notes on data:
  * ETF tickers below are placeholders. Most ETFs start 2000-2011, so any "old portfolio"
    sim needs index/factor proxies (Fama-French, MSCI/Shiller series, FRED T-bill/Treasury
    yields) before ETF inception. Swap the TICKERS map freely.
  * If you price from ETF adjusted closes, expense ratios are already embedded. Do not
    subtract them again.
  * Tax parameters (IRA limit, brackets) change annually. Verify before trusting output.
"""
import calendar
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum, IntEnum

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class AccountType(Enum):
    TAXABLE = "taxable"      # individual brokerage
    ROTH_IRA = "roth_ira"    # after-tax in, tax-free growth/out
    TRAD_IRA = "trad_ira"    # deductible in, ordinary income out


class LotMethod(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    HIFO = "hifo"            # highest-cost-first, minimizes realized gains


class ContribMode(Enum):
    LUMP_SUM = "lump_sum"    # `amount` deposited once at start
    MONTHLY = "monthly"      # `amount` deposited every month
    SCHEDULE = "schedule"    # explicit [(date, amount), ...]; negatives = withdrawals


class Detail(IntEnum):
    """How deep the stock allocation tree goes. Finer fields are ignored below the level."""
    STOCK_BOND = 1           # only top-level weights; generic domestic/foreign + regions
    DOMESTIC_FOREIGN = 2     # + foreign_share_of_stocks
    REGIONS = 3              # + region_weights within foreign


class Rebalance(Enum):
    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class CoreIndex(Enum):
    """Domestic core vehicle when the style grid is off."""
    SP500 = "SP500"
    TOTAL_MARKET = "TOTAL_MARKET"
    NASDAQ100 = "NASDAQ100"
    RUSSELL2000 = "RUSSELL2000"
    DOW = "DOW"


class Preset(Enum):
    ALL_STOCK = "all_stock"
    SIXTY_FORTY = "60_40"
    EIGHTY_TWENTY = "80_20"
    NINETY_TBILL = "90_tbill_10"
    PERMANENT = "permanent_portfolio"
    STOCKS_LEAP_OVERLAY = "stocks_leap_overlay"


# ----------------------------------------------------------------------------
# Instrument universe (leaf key -> ticker). Replace with your own data sources.
# ----------------------------------------------------------------------------

TICKERS = {
    # cash / bonds
    "TBILL": "BIL", "BOND_SHORT_TSY": "SHY", "BOND_INT_TSY": "IEF", "BOND_LONG_TSY": "TLT",
    "BOND_TIPS": "TIP", "BOND_AGG": "AGG", "BOND_MUNI": "MUB",
    # domestic core index options
    "CORE_SP500": "VOO", "CORE_TOTAL_MARKET": "VTI", "CORE_NASDAQ100": "QQQ",
    "CORE_RUSSELL2000": "IWM", "CORE_DOW": "DIA",
    # domestic size x style grid
    "US_LARGE_VALUE": "VTV", "US_LARGE_BLEND": "VOO", "US_LARGE_GROWTH": "VUG",
    "US_MID_VALUE": "VOE", "US_MID_BLEND": "VO", "US_MID_GROWTH": "VOT",
    "US_SMALL_VALUE": "VBR", "US_SMALL_BLEND": "VB", "US_SMALL_GROWTH": "VBK",
    # foreign regions
    "EUROPE_DEV": "VGK", "PACIFIC_DEV": "VPL", "PACIFIC_EM": "EEMA", "OTHER_FOREIGN": "EWC",
    # precious metals
    "GOLD_PHYSICAL": "GLD", "SILVER_PHYSICAL": "SLV", "GOLD_MINERS": "GDX",
    "JUNIOR_MINERS": "GDXJ", "SILVER_MINERS": "SIL",
    # synthetic
    "LEAP_LADDER": "LEAP_LADDER",
}
SECTOR_TICKERS = {
    "TECH": "XLK", "FINANCIALS": "XLF", "HEALTH": "XLV", "ENERGY": "XLE", "DISCRETIONARY": "XLY",
    "STAPLES": "XLP", "INDUSTRIALS": "XLI", "MATERIALS": "XLB", "UTILITIES": "XLU",
    "REAL_ESTATE": "XLRE", "COMM": "XLC",
}
for _s, _t in SECTOR_TICKERS.items():
    TICKERS["SECTOR_" + _s] = _t

# tax tags
COLLECTIBLE_LEAVES = {"GOLD_PHYSICAL", "SILVER_PHYSICAL"}  # grantor-trust bullion ETFs: 28% max LT
TREASURY_LEAVES = {"TBILL", "BOND_SHORT_TSY", "BOND_INT_TSY", "BOND_LONG_TSY", "BOND_TIPS"}
MUNI_LEAVES = {"BOND_MUNI"}

DEFAULT_FOREIGN_SHARE = 0.40
DEFAULT_REGIONS = {"EUROPE_DEV": 0.45, "PACIFIC_DEV": 0.25, "PACIFIC_EM": 0.20, "OTHER_FOREIGN": 0.10}
METAL_KEYS = {"GOLD_PHYSICAL", "SILVER_PHYSICAL", "GOLD_MINERS", "JUNIOR_MINERS", "SILVER_MINERS"}


def _normalize(d):
    s = sum(d.values())
    if s <= 0 or any(v < 0 for v in d.values()):
        raise ValueError(f"weights must be non-negative with positive sum: {d}")
    return {k: v / s for k, v in d.items()}


# ----------------------------------------------------------------------------
# Time / costs / rebalancing
# ----------------------------------------------------------------------------


@dataclass
class TimeConfig:
    start: date = date(2010, 1, 1)
    end: date = date(2025, 12, 31)
    benchmark: str = "VTI"
    reinvest_dividends: bool = True
    inflation_adjust: bool = False       # report real (CPI-deflated) values
    liquidate_at_end: bool = True        # apply terminal_tax so account types compare after-tax
    rebalance: Rebalance = Rebalance.ANNUAL
    rebalance_band: float | None = None  # e.g. 0.05 = rebalance when any sleeve drifts 5pts abs
    rebalance_with_cashflows_first: bool = True  # steer new money to underweights before selling


@dataclass
class CostConfig:
    commission_per_trade: float = 0.0
    slippage_bps: float = 1.0
    option_commission_per_contract: float = 0.65
    option_half_spread_pct: float = 0.01  # of premium, each side


# ----------------------------------------------------------------------------
# Tax rules
# ----------------------------------------------------------------------------


@dataclass
class TaxConfig:
    account_type: AccountType = AccountType.TAXABLE
    ordinary_rate: float = 0.24          # federal marginal during accumulation
    stcg_rate: float | None = None       # None -> ordinary_rate
    ltcg_rate: float = 0.15
    qualified_div_rate: float | None = None  # None -> ltcg_rate
    collectibles_rate: float = 0.28      # cap; applied as min(this, ordinary_rate)
    niit_rate: float = 0.038
    apply_niit: bool = False
    state_rate: float = 0.0              # 0 for no-income-tax states
    state_taxes_treasury_interest: bool = False
    ltcg_holding_days: int = 365         # long-term requires strictly more than this
    lot_method: LotMethod = LotMethod.HIFO
    model_wash_sale: bool = False
    tax_paid_from_portfolio: bool = True  # annual dividend/interest tax drag reduces holdings

    # IRA mechanics
    annual_ira_limit: float = 7500.0     # verify for the sim year; ignores catch-up/phaseouts
    enforce_ira_limit: bool = True
    overflow_to_taxable: bool = True     # excess contributions go to a taxable account
    trad_equalize_outlay: bool = False   # treat Trad contribution c as costing c*(1-ordinary_rate)
                                         # after-tax, so Roth vs Trad compares equal real outlay

    # Terminal/withdrawal
    retirement_ordinary_rate: float | None = None  # None -> ordinary_rate
    retirement_state_rate: float | None = None     # None -> state_rate
    early_withdrawal: bool = False       # pre-59.5 liquidation
    early_withdrawal_penalty: float = 0.10

    def validate(self):
        for n in ("ordinary_rate", "ltcg_rate", "collectibles_rate", "niit_rate", "state_rate"):
            if not 0 <= getattr(self, n) <= 1:
                raise ValueError(f"{n} must be in [0,1]")

    @property
    def is_taxable_account(self):
        return self.account_type is AccountType.TAXABLE

    def _add_on(self):
        return self.state_rate + (self.niit_rate if self.apply_niit else 0.0)

    def _st(self):
        return (self.stcg_rate if self.stcg_rate is not None else self.ordinary_rate) + self._add_on()

    def _lt(self):
        return self.ltcg_rate + self._add_on()

    def rate_gain(self, holding_days, collectible=False, section_1256=False):
        """Rate on a realized gain. Zero inside IRAs. 1256 contracts (SPX/NDX/XSP options): 60/40."""
        if not self.is_taxable_account:
            return 0.0
        if section_1256:
            return 0.6 * self._lt() + 0.4 * self._st()
        if holding_days <= self.ltcg_holding_days:
            return self._st()
        if collectible:
            return min(self.collectibles_rate, self.ordinary_rate) + self._add_on()
        return self._lt()

    def rate_dividend(self, qualified=True):
        if not self.is_taxable_account:
            return 0.0
        base = (self.qualified_div_rate if self.qualified_div_rate is not None else self.ltcg_rate) \
            if qualified else self.ordinary_rate
        return base + self._add_on()

    def rate_interest(self, leaf=None):
        """Interest tax. Treasuries skip state tax by default; munis exempt federally (ignores AMT)."""
        if not self.is_taxable_account or leaf in MUNI_LEAVES:
            return 0.0
        state = 0.0 if (leaf in TREASURY_LEAVES and not self.state_taxes_treasury_interest) else self.state_rate
        return self.ordinary_rate + state + (self.niit_rate if self.apply_niit else 0.0)

    def terminal_tax(self, balance, unrealized_lt=0.0, unrealized_st=0.0, contributions_basis=None):
        """Tax if the account were liquidated and withdrawn at the end of the sim."""
        if self.account_type is AccountType.TAXABLE:
            return unrealized_lt * self._lt() + unrealized_st * self._st()
        if self.account_type is AccountType.TRAD_IRA:
            r = (self.retirement_ordinary_rate if self.retirement_ordinary_rate is not None
                 else self.ordinary_rate)
            s = self.retirement_state_rate if self.retirement_state_rate is not None else self.state_rate
            return balance * (r + s + (self.early_withdrawal_penalty if self.early_withdrawal else 0.0))
        # Roth: qualified withdrawals are free; early withdrawal taxes/penalizes earnings only
        if self.early_withdrawal and contributions_basis is not None:
            earnings = max(balance - contributions_basis, 0.0)
            return earnings * (self.ordinary_rate + self.state_rate + self.early_withdrawal_penalty)
        return 0.0

    def split_contributions(self, flows):
        """Apply the annual IRA cap. Returns (accepted_to_account, overflow). Withdrawals pass through."""
        if self.is_taxable_account or not self.enforce_ira_limit:
            return list(flows), []
        used, ok, over = defaultdict(float), [], []
        for d, a in sorted(flows):
            if a <= 0:
                ok.append((d, a))
                continue
            room = max(self.annual_ira_limit - used[d.year], 0.0)
            take = min(a, room)
            used[d.year] += take
            if take > 0:
                ok.append((d, take))
            if a - take > 0 and self.overflow_to_taxable:
                over.append((d, a - take))
        return ok, over


# ----------------------------------------------------------------------------
# Contributions
# ----------------------------------------------------------------------------


@dataclass
class ContributionPlan:
    mode: ContribMode = ContribMode.MONTHLY
    amount: float = 500.0                 # LUMP_SUM: one deposit; MONTHLY: per-month deposit
    starting_balance: float = 0.0         # optional day-0 deposit in any mode
    monthly_day: int = 1
    annual_increase: float = 0.0          # step-up in monthly amount each year (0.03 = 3%)
    schedule: list = field(default_factory=list)  # [(date, amount), ...] for SCHEDULE mode

    def cash_flows(self, start, end):
        flows = []
        if self.starting_balance:
            flows.append((start, float(self.starting_balance)))
        if self.mode is ContribMode.LUMP_SUM:
            if self.amount:
                flows.append((start, float(self.amount)))
        elif self.mode is ContribMode.MONTHLY:
            y, m = start.year, start.month
            while True:
                d = date(y, m, min(self.monthly_day, calendar.monthrange(y, m)[1]))
                if d > end:
                    break
                if d >= start:
                    years = ((y - start.year) * 12 + (m - start.month)) // 12
                    flows.append((d, self.amount * (1 + self.annual_increase) ** years))
                m += 1
                if m > 12:
                    m, y = 1, y + 1
        else:
            flows += [(d, float(a)) for d, a in self.schedule if start <= d <= end]
        return sorted(flows)


# ----------------------------------------------------------------------------
# Portfolio sleeves
# ----------------------------------------------------------------------------


@dataclass
class StyleTilt:
    """Continuous size/value tilt on the domestic core. Replaces the single core index when enabled."""
    enabled: bool = False
    size_tilt: float = 0.0     # -1 (all large) .. 0 (cap-weight) .. +1 (heavy small cap)
    value_tilt: float = 0.0    # -1 (growth) .. 0 (blend) .. +1 (value)
    strength: float = 1.5      # exponential tilt strength; +1 value_tilt at 1.5 ~ 79% value
    base_size: dict = field(default_factory=lambda: {"LARGE": 0.72, "MID": 0.18, "SMALL": 0.10})

    def grid(self):
        S = {"LARGE": -1, "MID": 0, "SMALL": 1}
        V = {"VALUE": 1, "BLEND": 0, "GROWTH": -1}
        sz = _normalize({k: v * math.exp(self.strength * self.size_tilt * S[k])
                         for k, v in self.base_size.items()})
        st = _normalize({k: math.exp(self.strength * self.value_tilt * V[k]) for k in V})
        return {f"US_{a}_{b}": sz[a] * st[b] for a in sz for b in st}


@dataclass
class SectorSatellite:
    pct_of_domestic: float = 0.0               # carve-out from the domestic sleeve
    weights: dict = field(default_factory=dict)  # sector -> weight; empty = equal-weight all 11


@dataclass
class LeapLadder:
    """Staggered long-dated option ladder. Capital = premium budget; sleeve size is set by
    PortfolioConfig.leaps."""
    underlying: str = "SPY"
    option_type: str = "call"
    target_delta: float = 0.80            # deep ITM approximates leveraged stock with limited premium
    rungs: int = 4
    initial_dte: int = 730                # days to expiry at purchase
    roll_at_dte: int = 180                # close and rebuy a fresh initial_dte contract here
    rung_spacing_days: int | None = None  # None -> even spacing across the roll cycle
    cash_reserve_pct: float = 0.0         # share of sleeve kept in T-bills to fund rolls/drawdowns
    iv_mode: str = "flat"                 # "flat" or "realized_x_mult"
    flat_iv: float = 0.20
    iv_multiplier: float = 1.15           # used if iv_mode == "realized_x_mult"
    dividend_yield: float = 0.013
    index_option_1256: bool = False       # True for cash-settled index options (SPX/XSP/NDX): 60/40 tax

    def spacing_days(self):
        return self.rung_spacing_days or max((self.initial_dte - self.roll_at_dte) // self.rungs, 1)

    def initial_dtes(self):
        """DTE of each rung at t0, staggering expiries."""
        sp = self.spacing_days()
        return [self.initial_dte - i * sp for i in range(self.rungs)]

    def validate(self):
        if self.rungs < 1 or self.roll_at_dte >= self.initial_dte:
            raise ValueError("need rungs >= 1 and roll_at_dte < initial_dte")
        if not 0 < self.target_delta <= 1:
            raise ValueError("target_delta in (0,1]")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")


PRESETS = {
    Preset.ALL_STOCK: dict(stocks=1.0),
    Preset.SIXTY_FORTY: dict(stocks=0.60, bonds=0.40),
    Preset.EIGHTY_TWENTY: dict(stocks=0.80, bonds=0.20),
    Preset.NINETY_TBILL: dict(stocks=0.90, tbills=0.10),
    Preset.PERMANENT: dict(stocks=0.25, bonds=0.25, tbills=0.25, metals=0.25,
                           bond_mix={"BOND_LONG_TSY": 1.0}, metal_mix={"GOLD_PHYSICAL": 1.0}),
    Preset.STOCKS_LEAP_OVERLAY: dict(stocks=0.70, tbills=0.15, leaps=0.15),
}


@dataclass
class PortfolioConfig:
    detail: Detail = Detail.STOCK_BOND

    # 1. Top-level split (must sum to 1)
    stocks: float = 0.60
    bonds: float = 0.40
    tbills: float = 0.0
    metals: float = 0.0       # 4. precious-metals sleeve
    leaps: float = 0.0        # 6. LEAP ladder sleeve (premium budget)

    # 2/3. Foreign split and regions (gated by `detail`)
    foreign_share_of_stocks: float = DEFAULT_FOREIGN_SHARE
    region_weights: dict = field(default_factory=lambda: dict(DEFAULT_REGIONS))
    # US is the domestic leg; regions split the foreign leg.

    # Bond / metal internals
    bond_mix: dict = field(default_factory=lambda: {"BOND_AGG": 1.0})
    metal_mix: dict = field(default_factory=lambda: {
        "GOLD_PHYSICAL": 0.60, "SILVER_PHYSICAL": 0.15, "GOLD_MINERS": 0.15,
        "JUNIOR_MINERS": 0.05, "SILVER_MINERS": 0.05})

    # 5/7. Domestic vehicle choices
    core_us_index: CoreIndex = CoreIndex.TOTAL_MARKET
    sectors: SectorSatellite = field(default_factory=SectorSatellite)
    style: StyleTilt = field(default_factory=StyleTilt)

    leap: LeapLadder = field(default_factory=LeapLadder)

    @classmethod
    def from_preset(cls, preset, **overrides):
        return cls(**{**PRESETS[preset], **overrides})

    def validate(self):
        tops = [self.stocks, self.bonds, self.tbills, self.metals, self.leaps]
        if any(t < 0 for t in tops) or abs(sum(tops) - 1.0) > 1e-6:
            raise ValueError(f"top-level weights must be >=0 and sum to 1 (got {sum(tops):.6f})")
        if not 0 <= self.foreign_share_of_stocks <= 1:
            raise ValueError("foreign_share_of_stocks in [0,1]")
        if not set(self.region_weights) <= set(DEFAULT_REGIONS):
            raise ValueError(f"region keys must be in {set(DEFAULT_REGIONS)}")
        if not set(self.bond_mix) <= {k for k in TICKERS if k.startswith("BOND_")}:
            raise ValueError("unknown bond_mix key")
        if not set(self.metal_mix) <= METAL_KEYS:
            raise ValueError(f"metal_mix keys must be in {METAL_KEYS}")
        if not set(self.sectors.weights) <= set(SECTOR_TICKERS):
            raise ValueError(f"sector keys must be in {set(SECTOR_TICKERS)}")
        if not 0 <= self.sectors.pct_of_domestic <= 1:
            raise ValueError("pct_of_domestic in [0,1]")
        if not (-1 <= self.style.size_tilt <= 1 and -1 <= self.style.value_tilt <= 1):
            raise ValueError("tilts in [-1,1]")
        if self.leaps > 0:
            self.leap.validate()

    def resolve(self):
        """Flatten the tree to {leaf_key: weight} summing to 1."""
        self.validate()
        w = defaultdict(float)
        if self.stocks:
            f = (self.foreign_share_of_stocks if self.detail >= Detail.DOMESTIC_FOREIGN
                 else DEFAULT_FOREIGN_SHARE)
            regions = _normalize(self.region_weights) if self.detail >= Detail.REGIONS else DEFAULT_REGIONS
            dom, frn = self.stocks * (1 - f), self.stocks * f
            sat = dom * self.sectors.pct_of_domestic
            core = dom - sat
            if self.style.enabled:
                for k, v in self.style.grid().items():
                    w[k] += core * v
            else:
                w["CORE_" + self.core_us_index.name] += core
            if sat:
                sw = _normalize(self.sectors.weights or {s: 1.0 for s in SECTOR_TICKERS})
                for s, v in sw.items():
                    w["SECTOR_" + s] += sat * v
            for r, v in regions.items():
                w[r] += frn * v
        for k, v in _normalize(self.bond_mix).items():
            w[k] += self.bonds * v
        w["TBILL"] += self.tbills
        if self.metals:
            for k, v in _normalize(self.metal_mix).items():
                w[k] += self.metals * v
        if self.leaps:
            reserve = self.leaps * self.leap.cash_reserve_pct
            w["TBILL"] += reserve
            w["LEAP_LADDER"] += self.leaps - reserve
        return {k: v for k, v in w.items() if v > 1e-12}

    def to_tickers(self):
        return {TICKERS[k]: v for k, v in self.resolve().items()}

    def warnings(self):
        out = []
        if self.detail < Detail.DOMESTIC_FOREIGN and self.foreign_share_of_stocks != DEFAULT_FOREIGN_SHARE:
            out.append("detail<2: foreign_share_of_stocks ignored (using generic 40%).")
        if self.detail < Detail.REGIONS and self.region_weights != DEFAULT_REGIONS:
            out.append("detail<3: region_weights ignored (using generic split).")
        if self.style.enabled and self.core_us_index is not CoreIndex.TOTAL_MARKET:
            out.append("style grid on: core_us_index ignored.")
        return out


# ----------------------------------------------------------------------------
# Top-level bundle
# ----------------------------------------------------------------------------


@dataclass
class SimConfig:
    time: TimeConfig = field(default_factory=TimeConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)
    contributions: ContributionPlan = field(default_factory=ContributionPlan)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def validate(self):
        if self.time.end <= self.time.start:
            raise ValueError("end must be after start")
        self.tax.validate()
        self.portfolio.validate()

    def warnings(self):
        out = self.portfolio.warnings()
        p, t = self.portfolio, self.tax
        if p.leaps > 0 and t.is_taxable_account and not p.leap.index_option_1256:
            L = p.leap
            steady = L.initial_dte - L.roll_at_dte
            if steady <= t.ltcg_holding_days:
                out.append(f"LEAP rolls after {steady}d: gains are short-term in a taxable account. "
                           f"Use initial_dte - roll_at_dte > {t.ltcg_holding_days} or 1256 index options.")
            first = [d - L.roll_at_dte for d in L.initial_dtes()]
            if min(first) <= t.ltcg_holding_days < steady:
                out.append("First-cycle ladder rungs are rolled before the long-term threshold "
                           "(initial staggering); expect short-term gains in year one.")
        if p.metals and t.is_taxable_account and set(p.metal_mix) & COLLECTIBLE_LEAVES:
            out.append("Physical bullion ETFs taxed as collectibles (up to 28% LT) in taxable accounts; "
                       "miners are ordinary equities.")
        if not t.is_taxable_account and t.enforce_ira_limit:
            out.append(f"IRA limit ${t.annual_ira_limit:,.0f}/yr enforced; "
                       f"call tax.split_contributions(flows) for accepted vs overflow.")
        return out

    def summary(self):
        self.validate()
        print(f"Account: {self.tax.account_type.value} | {self.time.start} -> {self.time.end}")
        flows = self.contributions.cash_flows(self.time.start, self.time.end)
        print(f"Contributions: {len(flows)} flows, total ${sum(a for _, a in flows):,.0f}")
        print("Allocation:")
        for k, v in sorted(self.portfolio.resolve().items(), key=lambda kv: -kv[1]):
            print(f"  {k:<18}{TICKERS[k]:<12}{v:7.2%}")
        for w in self.warnings():
            print("  ! " + w)


if __name__ == "__main__":
    cfg = SimConfig(
        time=TimeConfig(start=date(2010, 1, 1), end=date(2025, 12, 31)),
        tax=TaxConfig(account_type=AccountType.TAXABLE, ordinary_rate=0.24, apply_niit=False),
        contributions=ContributionPlan(mode=ContribMode.MONTHLY, amount=500, annual_increase=0.03),
        portfolio=PortfolioConfig(
            detail=Detail.REGIONS, stocks=0.70, bonds=0.0, tbills=0.10, metals=0.05, leaps=0.15,
            foreign_share_of_stocks=0.35,
            style=StyleTilt(enabled=True, size_tilt=0.5, value_tilt=0.5),
            sectors=SectorSatellite(pct_of_domestic=0.10, weights={"TECH": 2, "HEALTH": 1}),
        ),
    )
    cfg.summary()
    print("\n-- same config, Roth IRA --")
    replace(cfg, tax=replace(cfg.tax, account_type=AccountType.ROTH_IRA)).summary()