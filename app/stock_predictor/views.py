from django.shortcuts import render, redirect
from django.http import JsonResponse
import sys
sys.path.append('../src/scripts')

import os
from run_arma import run_arma
from run_arima import run_arima
from run_lstm import run_lstm
from run_lstm_w_sent import run_lstm as run_lstm_w_sent
import datetime
import pickle

DATA_PATH = './data/'
PATH_TO_SENT_DATA = '../data/sentiment_data.csv'
def homepage(request):
    return render(request, 'homepage.html')


def get_result(request):
    if request.method == 'POST':

        ticker = request.POST['tickerSelect']

        today = datetime.datetime.today()
        files = os.listdir(DATA_PATH)
        ticker_file_name = ticker + str(today.date()) + '.pkl'
        found = False
        for file in files:
            if file == ticker_file_name:
                with open(DATA_PATH + file, 'rb') as f:
                    arma_prediction, arma_results = pickle.load(f)
                    arima_prediction, arima_results = pickle.load(f)
                    lstm_prediction, lstm_results = pickle.load(f)
                    lstm_w_sent_prediction, lstm_w_sent_results = pickle.load(f)
                    found = True
                    break
            
        if not found:
            arma_prediction, arma_results = run_arma(ticker, show_plot=False)
            arima_prediction, arima_results = run_arima(ticker, show_plot=False)
            lstm_prediction, lstm_results = run_lstm(ticker, show_plot=False)
            lstm_w_sent_prediction, lstm_w_sent_results = run_lstm_w_sent(
                ticker, show_plot=False, path_to_sent_data= PATH_TO_SENT_DATA)
            results = [arma_results, arima_results, lstm_results, lstm_w_sent_results]
            for result in results:
                for k,v in result.items():
                    result[k] = round(v, 2)

            with open(DATA_PATH + ticker_file_name, 'wb') as f:
                pickle.dump((arma_prediction, arma_results), f)
                pickle.dump((arima_prediction, arima_results), f)
                pickle.dump((lstm_prediction, lstm_results), f)
                pickle.dump((lstm_w_sent_prediction, lstm_w_sent_results), f)

        context = {'today_str':str(today.date()),
                   'arma_prediction': [round(arma_prediction, 2), arma_results],
                   'arima_prediction': [round(arima_prediction, 2), arima_results],
                   'lstm_prediction': [round(lstm_prediction, 2), lstm_results],
                   'lstm_w_sent_prediction': [round(lstm_w_sent_prediction, 2), lstm_w_sent_results]}

        return JsonResponse(context)

def report(request):
    return render(request, 'report.html')