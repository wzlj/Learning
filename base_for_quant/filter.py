def filter_specials(context, n=60):
    # type: (Context, int) -> list
    """
    过滤掉：1）三停：涨停、跌停、停牌；2）三特：st, *st, 退；3）科创、创业; 4）次新；
    适用于开盘前选股，如果是盘中，用curr_data[security].last_price替代curr_data[stock].day_open
    :param n: 获取n个交易日之前上市的股票
    """
    #     获取交易日
    trd_days = get_trade_days(end_date=end_dt, count=n)
    #     获取n个交易日之前前上市股票【即过滤掉次新股】
    stock_list = get_all_securities('stock', trd_days[0]).index.tolist()
    #
    curr_data = get_current_data()
    stock_list = [stock for stock in stock_list if not (
            (curr_data[stock].day_open == curr_data[stock].high_limit) or   # 涨停开盘
            (curr_data[stock].day_open == curr_data[stock].low_limit) or    # 跌停开盘
            curr_data[stock].paused or  # 停牌
            curr_data[stock].is_st or   # ST
            ('ST' in curr_data[stock].name) or
            ('*' in curr_data[stock].name) or
            ('退' in curr_data[stock].name) or
            (stock.startswith('300')) or    # 创业
            (stock.startswith('688'))   # 科创
    )]
    #
    return stock_list
