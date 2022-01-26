import warnings
warnings.filterwarnings("ignore")

import ta
import numpy as np

def add_ta_features(df): # Add technical analysis features
    windows = [6,18,24,30,50,100,200]

    for w in windows:
        if len(df) >= w:
            # RSI
            df['RSI_' + str(w)] = ta.momentum.RSIIndicator(df['Close'], window=w, fillna=True).rsi()
            # MACD
            for w2 in windows:
                if w > w2:
                    # Will utilize macd_diff because that is more normalized
                    df['MACD_f'+str(w2)+'_s'+str(w)] = ta.trend.MACD(df['Close'], window_slow=w2, window_fast=w, fillna=True).macd_diff()

            # Bollinger Bands
            ## Stdev default=2, but can change it if desired
            # Currently returning high/low band indicators, but can add actual values if desired.
            bbands = ta.volatility.BollingerBands(df['Close'], window=w, fillna=True)
            df['BBands_' + str(w) + '_h_ind'] = bbands.bollinger_hband_indicator()
            df['BBands_' + str(w) + '_l_ind'] = bbands.bollinger_lband_indicator()
            #actual values
            df['BBands_' + str(w) + 'hband'] = bbands.bollinger_hband()
            df['BBands_' + str(w) + 'lband'] = bbands.bollinger_lband()

            # Average True Range (ATR)
            df['ATR_' + str(w)] = ta.volatility.AverageTrueRange(
                high=df['High'],low=df['Low'],close=df['Close'], window=w, fillna=True).average_true_range()
                
            # Donchian Channel (DONCHIAN)
            d_channel = ta.volatility.DonchianChannel(
                high=df['High'],low=df['Low'],close=df['Close'], window=w, fillna=True)
            df['DONCHAIN_' + str(w) + 'hband'] = d_channel.donchian_channel_hband()
            df['DONCHAIN_' + str(w) + 'lband'] = d_channel.donchian_channel_lband()
                
            # Keltner Channel (KELTNER)
            # Using SMA as centerline
            k_channel = ta.volatility.KeltnerChannel(
                high=df['High'],low=df['Low'],close=df['Close'], window=w, 
                original_version= True, fillna=True)
            df['KELTNER_' + str(w) + '_h_ind'] = k_channel.keltner_channel_hband_indicator()
            df['KELTNER_' + str(w) + '_l_ind'] = k_channel.keltner_channel_lband_indicator()
                
            # Stochastic Oscillator (SR/STOCH)
            df['STOCH_' + str(w)] = ta.momentum.StochasticOscillator(
                high=df['High'],low=df['Low'],close=df['Close'], window=w, fillna=True).stoch()

            # Chaikin Money Flow Indicator (CMF)
            df['CMF_' + str(w)] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['High'],low=df['Low'],close=df['Close'],volume=df['Volume'], window=w, fillna=True).chaikin_money_flow()

            # Ichimoku Indicator (ICHI)
            for w2 in windows:
                for w3 in windows:
                    if (w > w2) & (w2 > w3):
                        ichimoku = ta.trend.IchimokuIndicator(
                            high=df['High'],low=df['Low'],window1=w3, window2=w2, window3=w, fillna=True)
                        df['ICHI_conv_' + str(w3)+'_'+str(w2)+'_'+str(w)] = ichimoku.ichimoku_conversion_line()
                        df['ICHI_base_' + str(w3)+'_'+str(w2)+'_'+str(w)] = ichimoku.ichimoku_base_line()
                        df['ICHI_diff_' + str(w3)+'_'+str(w2)+'_'+str(w)] = df['ICHI_conv_' + str(w3)+'_'+str(w2)+'_'+str(w)] - df['ICHI_base_' + str(w3)+'_'+str(w2)+'_'+str(w)]


                # SMA
                df['SMA_' + str(w)] = ta.trend.SMAIndicator(df['Close'], window=w, fillna=True).sma_indicator()

                # SMA Crossover
                for w2 in windows:
                    if w > w2:
                        sma_s = ta.trend.SMAIndicator(df['Close'], window=w, fillna=True).sma_indicator()
                        sma_f = ta.trend.SMAIndicator(df['Close'], window=w2, fillna=True).sma_indicator()
                        df['SMA_cross_f' + str(w2) + '_s' + str(w)] = sma_f - sma_s

                # EMA
                df['EMA_' + str(w)] = ta.trend.EMAIndicator(df['Close'], window=w, fillna=True).ema_indicator()

                # EMA Crossover
                for w2 in windows:
                    if w > w2:
                        ema_s = ta.trend.EMAIndicator(df['Close'], window=w, fillna=True).ema_indicator()
                        ema_f = ta.trend.EMAIndicator(df['Close'], window=w2, fillna=True).ema_indicator()
                        df['SMA_cross_f' + str(w2) + '_s' + str(w)] = ema_f - ema_s


            ## WINDOW NOT REQUIRED
            # On Balance Volume Indicator (OBV)
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['Close'],volume=df['Volume'], fillna=True).on_balance_volume()

            # Volume-Price Trend (VPT)
            df['VPT'] = ta.volume.VolumePriceTrendIndicator(
                close=df['Close'],volume=df['Volume'], fillna=True).volume_price_trend()

            # Accumulation/Distribution Index Indicator (ADI)
            df['ADI'] = ta.volume.AccDistIndexIndicator(
                high=df['High'],low=df['Low'],close=df['Close'],volume=df['Volume'], fillna=True).acc_dist_index()

        # Getting daily returns (pct and log) for 1,2,3 days
        return_days = [1,2,3]
        for day in return_days:
            df[f'{day}_day_return'] = (df['Close'] / df['Close'].shift(day)) - 1
            df[f'{day}_day_log_return'] = (np.log(df['Close']) - np.log(df['Close'].shift(day)) )* 100
        for day in return_days:
            df[f'{day}_day_return'].fillna(0, inplace=True)
            df[f'{day}_day_log_return'].fillna(0, inplace=True)

    return df.copy()