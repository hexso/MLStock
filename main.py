from stock_data_parse import *
from models.GRU import GRURapid




if __name__ == '__main__':
    stock_data = get_stock_data('005930')

    model = GRURapid()
    model.learn(model, stock_data)