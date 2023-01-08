import FinanceDataReader as fdr
import talib
import pandas as pd


def cal_stock_indicator(data):
    x = data
    x.columns = map(str.lower, x.columns)
    x['obv'] = talib.OBV(x['close'], volume=x['volume'])
    x = x.dropna()
    return x


def get_stock_data(stock_code: str, indicator=True):
    data = fdr.DataReader(stock_code)
    return cal_stock_indicator(data)



if __name__ == '__main__':
    data = get_stock_data('005930').dropna()
    change_func = lambda x: 0 if x < 15 / 100 else 1
    data['change_over'] = data['change'].apply(change_func)
    data['change_over_tmw'] = data['change_over'].shift(-1)
    data['indicator'] = 0
    print(data)
    for idx, val in enumerate(data['change_over']):
        for i in range(10):
            if idx + i >= len(data):
                break
            if data['change_over'][idx + i] == 1:
                data['indicator'][idx] = 1 - (0.1*idx)
                break

    print(data)
