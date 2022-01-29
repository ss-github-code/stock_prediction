import sys
import numpy as np
import pandas as pd
import datetime
import requests
import time
import json

from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PATH_TO_SENT_DATA = '../../data/sentiment_data.csv'
SLEEP_DURATION = 7
'''
We need to update the sentiment from today's news articles before we can run the modeling for predicting next day's high.
We use New York Times Developer API to fetch the news articles from the Business section. There is a rate limit on fast we 
can access the data using the free API (10 articles/min).
We need to also deal with the fact the API returns 10 articles at a time; so we need to make multiple calls to finish
processing all the news articles for today.
For each news article's headline and its corresponding lead paragraph, we use VADER sentiment intensity analyzer to 
get its sentiment scores (positive, negative, neutral, compound).

VADER: Valence  Aware  Dictionary  for sEntiment Reasoning: a simple rule-based model for general sentiment  analysis.
VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text
https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399
"We find that incorporating these heuristics improves the accuracy of the sentiment analysis engine across several domain contexts 
(social media text, NY Times  editorials,  movie  reviews,  and  product  reviews)."
We save the set of average scores across all articles in the day and update the sentiment data's csv file.
'''
def find_news_articles(begindate, nytimes_section, api_key):
    base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?'
    facet_str = f'&facet=true&begin_date={begindate}&end_date={begindate}'
    
    page = 0
    count = 0
    
    ret_list = []
    while True:
        url = base_url+nytimes_section+facet_str+f'&page={page}'+api_key
        r = requests.get(url)
        if r.status_code != 200:
            print(r.status_code)
        data = json.loads(r.content)
        time.sleep(SLEEP_DURATION)
        if page == 0:
            tot_articles = data['response']['meta']['hits']
            print(begindate, nytimes_section, 'tot_articles', tot_articles)
        for i, doc in enumerate(data['response']['docs']):
            ret_list.append((doc['headline']['main'], doc['lead_paragraph'], doc['web_url']))
            count += 1
        if count >= tot_articles:
            break
        page += 1
    #print(len(ret_list))
    return ret_list

def update_sentiment_data(api_key):
    api_key = f'&api-key={api_key}'
    sent_data = pd.read_csv(PATH_TO_SENT_DATA)
    sent_data['Date'] = pd.to_datetime(sent_data['Date'])
    sent_data.sort_values(['Date'], inplace=True, ignore_index=True)
    # begin_date is the last day for which we have the sentiment data for.
    begin_date = sent_data.iloc[-1]['Date'] + datetime.timedelta(days=1)
    today = datetime.datetime.today()

    print(begin_date, today)

    daily_sentiment = defaultdict(defaultdict)
    sid_obj = SentimentIntensityAnalyzer()

    while begin_date <= today:

        days_sentiment_pos, days_sentiment_neg, days_sentiment_neu, days_sentiment_comp = 0, 0, 0, 0
        date_str = str(begin_date.year) + str(begin_date.month).zfill(2) + str(begin_date.day).zfill(2)

        news_desk_str = 'fq=news_desk:("Financial" "Business" "Business Day")'
        section_str = 'fq=section_name:("Your Money" "Business" "Business Day")'

        news_desk_list = find_news_articles(date_str, news_desk_str, api_key)
        section_list = find_news_articles(date_str, section_str, api_key)

        final_urls = set()
        for news in news_desk_list: # tuple of 3: headline, lead_paragraph, web_url
            if news[2] not in final_urls:
                # print('adding news desk article', news[0])
                final_urls.add(news[2])
                sentiment_dict = sid_obj.polarity_scores(news[0] + news[1]) # use VADER sentiment analyzer on the headline & leading paragraph
                days_sentiment_pos += sentiment_dict['pos']
                days_sentiment_neg += sentiment_dict['neg']
                days_sentiment_neu += sentiment_dict['neu']
                days_sentiment_comp += sentiment_dict['compound']
        for news in section_list: # tuple of 3: headline, lead_paragraph, web_url
            if news[2] not in final_urls:
                # print('adding section article', news[0])
                final_urls.add(news[2])
                sentiment_dict = sid_obj.polarity_scores(news[0] + news[1])
                days_sentiment_pos += sentiment_dict['pos']
                days_sentiment_neg += sentiment_dict['neg']
                days_sentiment_neu += sentiment_dict['neu']
                days_sentiment_comp += sentiment_dict['compound']

        num_news_items = len(final_urls)
        if num_news_items > 0:
            daily_sentiment[date_str]['pos'] = days_sentiment_pos/num_news_items
            daily_sentiment[date_str]['neg'] = days_sentiment_neg/num_news_items
            daily_sentiment[date_str]['neu'] = days_sentiment_neu/num_news_items
            daily_sentiment[date_str]['compound'] = days_sentiment_comp/num_news_items
        else:
            daily_sentiment[date_str]['pos'] = 0
            daily_sentiment[date_str]['neg'] = 0
            daily_sentiment[date_str]['neu'] = 0
            daily_sentiment[date_str]['compound'] = 0
        begin_date += datetime.timedelta(days=1)

    if len(daily_sentiment):
        new_df = pd.DataFrame(daily_sentiment).T
        new_df.reset_index(inplace=True)
        new_df.rename(columns={'index' : 'Date'}, inplace=True)
        new_df['Date'] = pd.to_datetime(new_df['Date'], format="%Y%m%d")
        # concatenate to the existing data frame and save the dataframe
        new_df = pd.concat([sent_data, new_df], axis=0, ignore_index=True)

        new_df.to_csv(PATH_TO_SENT_DATA, index=False)

if __name__ == '__main__':
    update_sentiment_data(sys.argv[1])