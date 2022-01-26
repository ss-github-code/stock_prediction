import requests
import pandas as pd

def get_quarterly_reports(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    fundamental_data = r.json()

    df = pd.DataFrame(fundamental_data['quarterlyReports'])
    df.drop(columns=['reportedCurrency','investmentIncomeNet', 
                    'netInterestIncome', 'interestIncome', 'interestExpense', 'nonInterestIncome', 
                    'depreciation','depreciationAndAmortization'], inplace=True)
    for col in df.columns:
        if col != 'fiscalDateEnding':
            df[col] = df[col].astype('int64')

    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df.sort_values(['fiscalDateEnding'], inplace=True)
    return df