#Interpreting.md

Interpreting result summaries from tool.

Config Summary
1. Account: 
    a. Taxable: Dividends and interest are taxed the day they're paid. Gains are taxed whenever something is sold (rebalancing, LEAP rolls, withdrawals). At the end, unrealized gains are taxed as if you sold everything.
    b. Roth IRA: no tax. You can turn on early_withdrawals=True in order to tax and penalize the earnings at the end.
    c. Traditional IRA: no tax. At the end, the whole balance is taxed at retirement_ordinary_rate + state + 10% penality if early_withdrawal is True.
2. Contributions: A few options
    a. LUMP_SUM: one deposit on day one
    b. MONTHLY: Once a month on monthly_day, with an optional yearly raise through annual_incrase
    c. SCHEDULE: any dates you list, so daily, biweekyl, or random dates work. 
    d. starting_balance: adds a day-one deposit on top of any mode
3. Allocation: 

Portfolio v. Benchmark
4. Outlay: Out-of-pocket money that was put in
5. Final Value (pre-liquidation): Account Balance on Last day
6. Liquidation Tax: Taxes for liquidating
7. Final Value after tax: Final Value - Tax
8. After-tax IRR (Money-weighted): Internal Rate of Return, Nominal metric (pre-inflation), that essentially is showing annualized return BUT accounting for "when" each contribution entered (supposed to handle a large number of 'irregular' cash flows)
9. CAGR (time-weighted, pre-liq.): Compound Annual Growth Rate, aka Time-Weighted Return (TWR), removing the distorting effect of external cash flows, this ISOLATES the pure performance of the underlying assets
10. Volatility: How much it swings per year, standard deviation of dailt returns.
11. Sharpe Ratio: Metric evaluates investement's return compared to risk. Calculated as R_p - R_f / (sigma), where R_p is investment return, R_f is risk-free rate (T-bill), and sigma is standard deviation of investment's excess return (volatility). Under 1 is suboptimal, 1-2 is good, 2-3 is very good, above 3 is excellent.
12. Max drawdown: Worst peak-to-bottom drop
13. Taxes Paid: Gains, dividend, interest taxes
14. Turnover: How much you have to sell to rebalance (you'll find a single fund will have 0%)
15. Trading Costs: 1 basis point of slippage is applied for each ETF trade. $0 comission by default, for options, 65 cents per contract plus a 1% half-spread, dividend reinvetment buys are free, expense ratios are built into real ETF prices

Then a portfolio summary, where it shows the difference in value, taxes upon termination required (when you withdraw) and also taxes on any gains (applicable for taxable brokerage account)