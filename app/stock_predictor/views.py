from django.shortcuts import render, redirect
from django.http import JsonResponse
import sys
sys.path.append('../src/scripts')
import os
from run_arma import run_arma
from run_arima import run_arima
from run_lstm import run_lstm
import datetime
import pickle

DATA_PATH = './data/'
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
                    found = True
                    break
            
        if not found:
            arma_prediction, arma_results = run_arma(ticker, show_plot=False)
            arima_prediction, arima_results = run_arima(ticker, show_plot=False)
            lstm_prediction, lstm_results = run_lstm(ticker, show_plot=False)
            with open(DATA_PATH + ticker_file_name, 'wb') as f:
                pickle.dump((arma_prediction, arma_results), f)
                pickle.dump((arima_prediction, arima_results), f)
                pickle.dump((lstm_prediction, lstm_results), f)

        context = {'today_str':str(today.date()),
                   'arma_prediction': [arma_prediction, arma_results],
                   'arima_prediction': [arima_prediction, arima_results],
                   'lstm_prediction': [lstm_prediction, lstm_results]}

        return JsonResponse(context)

def report(request):
    return render(request, 'report.html')