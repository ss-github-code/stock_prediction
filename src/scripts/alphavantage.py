import requests
import pandas as pd

'''
Alpha Vantage provides set of fundamental data APIs in various temporal dimensions covering key financial metrics, 
income statements, balance sheets, cash flow, and other fundamental data points.
We use the API to download quarterly reports for a given stock ticker symbol.
'''
def get_quarterly_reports(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    fundamental_data = r.json()

    df = pd.DataFrame(fundamental_data['quarterlyReports'])
    # columns of interest:
    # fiscalDateEnding, grossProfit, totalRevenue, costOfRevenue, costofGoodsAndServicesSold
    # operatingIncome, sellingGeneralAndAdministrative, researchAndDevelopment, operatingExpenses,
    # otherNonOperatingIncome, incomeBeforeTax, incomeTaxExpense, interestAndDebtExpense, 
    # netIncomeFromContinuingOperations, comprehensiveIncomeNetOfTax, ebit, ebitda, netIncome
    #
    # drop some of the columns like "USD", investmentIncome related, and depreciation related
    df.drop(columns=['reportedCurrency','investmentIncomeNet', 
                    'netInterestIncome', 'interestIncome', 'interestExpense', 'nonInterestIncome', 
                    'depreciation','depreciationAndAmortization'], inplace=True)
    for col in df.columns:
        if col != 'fiscalDateEnding':
            df[col] = df[col].astype('int64')

    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
    df.sort_values(['fiscalDateEnding'], inplace=True)
    return df