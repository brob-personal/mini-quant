from datetime import date
from portfolio_levers import *
from portfolio_engine import synthetic_data, load_yahoo, backtest, compare_account_types

cfg = SimConfig(
    time=TimeConfig(start=date(2012, 1, 1), end=date(2024, 12, 31), benchmark="VTI"),
    tax=TaxConfig(account_type=AccountType.ROTH_IRA),
    contributions=ContributionPlan(mode=ContribMode.MONTHLY, amount=500),
    portfolio=PortfolioConfig(stocks=0.80, bonds=0.20),
)

cfg.summary()                      # sanity check the allocation first
data = synthetic_data(cfg)         # or load_yahoo(cfg) with internet + yfinance
res = backtest(cfg, data)
res["portfolio"].summary()
res["benchmark"].summary()
print(compare_account_types(cfg, data).round(3))