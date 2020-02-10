import numpy as np

# ------------------------------------ define functions --------------------------------------------------------------
def generate_monthscore(inp):
    yr, mth = str(inp.year), str(inp.month)
    if len(mth) == 1:
        mth = '0' + mth
    return yr + '-' + mth


def subtract_month(inp):
    yr, mth = inp.year, inp.month - 1
    if mth == 0:
        yr -= 1
        mth += 12
    yr, mth = str(yr), str(mth)
    if len(mth) == 1:
        mth = '0' + mth
    return yr + mth


def generate_lag(inp, lag):
    x = []
    y = []
    for i in range(len(inp) - lag):
        x.append(inp.values[i:i + lag, 0])
        y.append(inp.values[i + lag, 0])
    return np.array(x), np.array(y)


def randomforest_predict(inp, lag, model, n_periods):
    inp = inp.values[-lag:].reshape(1, -1)
    res = []
    for i in range(n_periods):
        opt = model.predict(inp)
        res.append(opt)
        opt = opt.reshape(1, -1)
        inp = np.hstack([inp[:, 1:], opt])
    return np.array(res).reshape(-1, )
