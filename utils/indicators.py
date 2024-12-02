import numpy as np
import pandas as pd


def calculate_indicators(data):
    data = ind_williams_percent_r(data,14)
    data = ind_roc(data,14)
    data = ind_rsi(data,7)
    data = ind_rsi(data,14)
    data = ind_rsi(data,28)
    data = ind_macd(data, 8, 21)
    data = ind_bbands(data,20)
    data = ind_ichimoku_cloud(data)
    data = ind_ema(data, 3)
    data = ind_ema(data, 8)
    data = ind_ema(data, 15)
    data = ind_ema(data, 50)
    data = ind_ema(data, 100)
    data = ind_adx(data, 14)
    data = ind_donchian(data, 10)
    data = ind_donchian(data, 20)
    data = ind_alma(data, 10)
    data = ind_tsi(data, 13, 25)
    data = ind_zscore(data, 20)
    data = ind_log_return(data, 10)
    data = ind_log_return(data, 20)
    data = ind_vortex(data, 7)
    data = ind_aroon(data, 16)
    data = ind_ebsw(data, 14)
    data = ind_accbands(data, 20)
    data = ind_short_run(data, 14)
    data = ind_bias(data, 26)
    data = ind_ttm_trend(data, 5, 20)
    data = ind_percent_return(data, 10)
    data = ind_percent_return(data, 20)
    data = ind_kurtosis(data, 5)
    data = ind_kurtosis(data, 10)
    data = ind_kurtosis(data, 20)
    data = ind_eri(data, 13)
    data = ind_atr(data, 14)
    data = ind_keltner_channels(data, 20)
    data = ind_chaikin_volatility(data, 10)
    data = ind_stdev(data, 5)
    data = ind_stdev(data, 10)
    data = ind_stdev(data, 20)
    data = ta_vix(data, 21)
    data = ind_obv(data, 10)
    data = ind_chaikin_money_flow(data, 5)
    data = ind_volume_price_trend(data, 7)
    data = ind_accumulation_distribution_line(data, 3)
    data = ind_ease_of_movement(data, 14)
    return data


# Williams %R
def ind_williams_percent_r(data, window=14):
    highest_high = data["High"].rolling(window=window).max()
    lowest_low = data["Low"].rolling(window=window).min()
    data["Williams_%R{}".format(window)] = -((highest_high - data["Close"]) / (highest_high - lowest_low)) * 100
    return data


# Rate of Change
def ind_roc(data, window=14):
    data["ROC_{}".format(window)] = (data["Close"] / data["Close"].shift(window) - 1) * 100
    return data


# RSI
def ind_rsi(data, window=14) :
    delta = data["Close"].diff(1)
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return data


# MACD
def ind_macd(data, short_window=8, long_window=21, signal_window=9):
    short_ema = data["Close"].ewm(span = short_window, adjust = False).mean()
    long_ema = data["Close"].ewm(span = long_window, adjust = False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    data["MACD_Line"] = macd_line
    data["Signal_Line"] = signal_line
    data["MACD_Histogram"] = macd_histogram
    return data


# Bollinger Bands
def ind_bbands(data, window=20, num_std_dev=2) :
    data["midlle_band"] = data["Close"].rolling(window=window).mean()
    data["std"] = data["Close"].rolling(window=window).std()
    data["upper_band{}".format(window)] = data["midlle_band"] + (num_std_dev * data["std"])
    data["lower_band{}".format(window)] = data["midlle_band"] - (num_std_dev * data["std"])
    data.drop(["std"], axis=1, inplace=True)
    return data


# Ichimoku Cloud
def ind_ichimoku_cloud(data, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    tenkan_sen = (data["Close"].rolling(window=window_tenkan).max() + data["Close"].rolling(window=window_tenkan).min()) / 2
    kijun_sen = (data["Close"].rolling(window=window_kijun).max() + data["Close"].rolling(window=window_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (data["Close"].rolling(window=window_senkou_span_b).max() + data["Close"].rolling(window=window_senkou_span_b).min()) / 2
    chikou_span = data["Close"].shift(-window_chikou)
    data["Tenkan_sen"] = tenkan_sen
    data["Kijun_sen"] = kijun_sen
    data["Senkou_Span_A"] = senkou_span_a
    data["Senkou_Span_B"] = senkou_span_b
    data["Chikou_Span"] = chikou_span
    return data


# Moving Average (EMA)
def ind_ema(data, window=8):
    data["ema_{}".format(window)] = data["Close"].ewm(span=window, adjust=False).mean()
    return data


# ADX
def ind_adx(data, window=14): #14
    data["TR"] = abs(data["High"] - data["Low"]).combine_first(abs(data["High"] - data["Close"].shift(1))).combine_first(abs(data["Low"] - data["Close"].shift(1)))
    data["DMplus"] = (data["High"] - data["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    data["DMminus"] = (data["Low"].shift(1) - data["Low"]).apply(lambda x: x if x > 0 else 0)
    data["ATR"] = data["TR"].rolling(window=window).mean()
    data["DIplus"] = (data["DMplus"].rolling(window=window).mean() / data["ATR"]) * 100
    data["DIminus"] = (data["DMminus"].rolling(window=window).mean() / data["ATR"]) * 100
    data["DX"] = abs(data["DIplus"] - data["DIminus"]) / (data["DIplus"] + data["DIminus"]) * 100
    data["ADX_{}".format(window)] = data["DX"].rolling(window=window).mean()
    data.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return data


# Donchian Channel
def ind_donchian(data, window=10):
    highest_high = data["Close"].rolling(window=window).max()
    lowest_low = data["Close"].rolling(window=window).min()
    data["Donchian_Upper_{}".format(window)] = highest_high
    data["Donchian_Lower_{}".format(window)] = lowest_low
    return data


# Arnaud Legoux Moving Average (ALMA)
def ind_alma(data, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(data["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    data["ALMA_{}".format(window)] = alma_values
    return data


# True Strength Index (TSI)
def ind_tsi(data, short_period=13, long_period=25):
    price_diff = data["Close"].diff(1)
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    data["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return data


# Z-Score
def ind_zscore(data, window=20):
    rolling_mean = data["Close"].rolling(window=window).mean()
    rolling_std = data["Close"].rolling(window=window).std()
    z_score = (data["Close"] - rolling_mean) / rolling_std
    data["Z_Score_{}".format(window)] = z_score
    return data


# Log Return
def ind_log_return(data, window=5):
    data["LogReturn_{}".format(window)] = data["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return data


# Vortex Indicator
def ind_vortex(data, window=7):
    high_low = data["High"] - data["Low"]
    high_close_previous = abs(data["High"] - data["Close"].shift(1))
    low_close_previous = abs(data["Low"] - data["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    positive_vm = abs(data["High"].shift(1) - data["Low"])
    negative_vm = abs(data["Low"].shift(1) - data["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    data["Positive_VI_{}".format(window)] = positive_vi
    data["Negative_VI_{}".format(window)] = negative_vi
    return data


# Aroon Indicator
def ind_aroon(data, window=16):
    high_prices = data["High"]
    low_prices = data["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    data["Aroon_Up_{}".format(window)] = aroon_up
    data["Aroon_Down_{}".format(window)] = aroon_down
    return data


# Elder"s Bull Power e Bear Power
def ind_ebsw(data, window=14):
    ema = data["Close"].ewm(span=window, adjust=False).mean()
    bull_power = data["High"] - ema
    bear_power = data["Low"] - ema
    data["Bull_Power_{}".format(window)] = bull_power
    data["Bear_Power_{}".format(window)] = bear_power
    return data


# Acceleration Bands
def ind_accbands(data, window=20, acceleration_factor=0.02):
    sma = data["Close"].rolling(window=window).mean()
    band_difference = data["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    data["Upper_Band_{}".format(window)] = upper_band
    data["Lower_Band_{}".format(window)] = lower_band
    data["Middle_Band_{}".format(window)] = sma
    return data


# Short Run
def ind_short_run(data, window=14):
    short_run = data["Close"] - data["Close"].rolling(window=window).min()
    data["Short_Run_{}".format(window)] = short_run
    return data


# Bias
def ind_bias(data, window=26):
    moving_average = data["Close"].rolling(window=window).mean()
    bias = ((data["Close"] - moving_average) / moving_average) * 100
    data["Bias_{}".format(window)] = bias
    return data


# TTM Trend
def ind_ttm_trend(data, short_window=5, long_window=20):
    short_ema = data["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    data["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return data


# Percent Return
def ind_percent_return(data, window=1):
    percent_return = data["Close"].pct_change().rolling(window=window).mean() * 100
    data["Percent_Return_{}".format(window)] = percent_return
    return data


# Kurtosis
def ind_kurtosis(data, window=20):
    data["kurtosis_{}".format(window)] = data["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return data


# Elder's Force Index (ERI)
def ind_eri(data, window=13):
    price_change = data["Close"].diff()
    force_index = price_change * data["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    data["ERI_{}".format(window)] = eri
    return data


# ATR
def ind_atr(data, window=14):
    data["High-Low"] = data["High"] - data["Low"]
    data["High-PrevClose"] = abs(data["High"] - data["Close"].shift(1))
    data["Low-PrevClose"] = abs(data["Low"] - data["Close"].shift(1))
    data["TrueRange"] = data[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    data["atr_{}".format(window)] = data["TrueRange"].rolling(window=window, min_periods=1).mean()
    data.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return data


# Keltner Channels
def ind_keltner_channels(data, period=20, multiplier=2):
    data["TR"] = data.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    data["ATR"] = data["TR"].rolling(window=period).mean()
    data["Middle Band"] = data["Close"].rolling(window=period).mean()
    data["Upper Band"] = data["Middle Band"] + multiplier * data["ATR"]
    data["Lower Band"] = data["Middle Band"] - multiplier * data["ATR"]
    return data


# Chaikin Volatility
def ind_chaikin_volatility(data, window=10):
    daily_returns = data["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    data["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return data


# Standard Deviation
def ind_stdev(data, window=1):
    stdev_column = data["Close"].rolling(window=window).std()
    data["Stdev_{}".format(window)] = stdev_column
    return data


# Volatility Index (VIX)
def ta_vix(data, window=21):
    returns = data["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100
    data["VIX_{}".format(window)] = vix
    return data


# On-Balance Volume (OBV)
def ind_obv(data, window=10):
    price_changes = data["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (data["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    data["OBV_{}".format(window)] = obv_smoothed
    return data


# Chaikin Money Flow (CMF)
def ind_chaikin_money_flow(data, window=10):
    mf_multiplier = ((data["Close"] - data["Close"].shift(1)) + (data["Close"] - data["Close"].shift(1)).abs()) / 2
    mf_volume = mf_multiplier * data["Volume"]
    adl = mf_volume.cumsum()
    cmf = adl.rolling(window=window).mean() / data["Volume"].rolling(window=window).mean()
    data["CMF_{}".format(window)] = cmf
    return data


# Volume Price Trend (VPT)
def ind_volume_price_trend(data, window=10):
    price_change = data["Close"].pct_change()
    vpt = (price_change * data["Volume"].shift(window)).cumsum()
    data["VPT_{}".format(window)] = vpt
    return data


# Accumulation/Distribution Line
def ind_accumulation_distribution_line(data, window=10):
    money_flow_multiplier = ((data["Close"] - data["Close"].shift(1)) - (data["Close"].shift(1) - data["Close"])) / (data["Close"].shift(1) - data["Close"])
    money_flow_volume = money_flow_multiplier * data["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    data["A/D Line_{}".format(window)] = ad_line_smoothed
    return data


# Ease of Movement (EOM)
def ind_ease_of_movement(data, window=14):
    midpoint_move = ((data["High"] + data["Low"]) / 2).diff(1)
    box_ratio = data["Volume"] / 1000000 / (data["High"] - data["Low"])
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    data["EOM_{}".format(window)] = eom_smoothed
    return data


