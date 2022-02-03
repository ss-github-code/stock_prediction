from django.shortcuts import render, redirect
from django.http import JsonResponse
import sys
sys.path.append('../src/scripts')

import tempfile

import datetime
import pickle
import boto3
import os

S3 = 's3'
S3_REGION = 'us-west-2'
S3_BUCKET = 'miidata'
AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']

def homepage(request):
    return render(request, 'homepage.html')

def get_result(request):
    if request.method == 'POST':

        ticker = request.POST['tickerSelect']
        s3client = boto3.client(S3, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=S3_REGION)
        objs = s3client.list_objects(Bucket=S3_BUCKET)

        latest = None
        for obj in objs['Contents']:
            objName = obj['Key']

            sp = objName.split('-')
            if sp[0] == ticker:
                yr = int(sp[1])
                m = int(sp[2])
                d = int(sp[3].split('.')[0])
                yr, m, d
                ymd = datetime.datetime(year=yr, month=m, day=d)                
                if latest is None:
                    latest_obj = obj['Key']
                    latest = ymd
                elif ymd > latest:
                    latest_obj = obj['Key']
                    latest = ymd

        s3 = boto3.resource(S3, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=S3_REGION)
        bucket = s3.Bucket(S3_BUCKET)
        object = bucket.Object(latest_obj)

        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'wb') as f:
            object.download_fileobj(f)

        with open(tmp.name, 'rb') as f:
            arma_prediction, arma_results = pickle.load(f)
            arima_prediction, arima_results = pickle.load(f)
            lstm_prediction, lstm_results = pickle.load(f)
            lstm_w_sent_prediction, lstm_w_sent_results = pickle.load(f)
        
        results = [arma_results, arima_results, lstm_results, lstm_w_sent_results]
        for result in results:
            for k,v in result.items():
                result[k] = round(v, 2)

        context = {'today_str':str(latest.date()),
                   'arma_prediction': [round(arma_prediction, 2), arma_results],
                   'arima_prediction': [round(arima_prediction, 2), arima_results],
                   'lstm_prediction': [round(lstm_prediction, 2), lstm_results],
                   'lstm_w_sent_prediction': [round(lstm_w_sent_prediction, 2), lstm_w_sent_results]}

        return JsonResponse(context)

def report(request):
    return render(request, 'report.html')