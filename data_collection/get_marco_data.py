from fredapi import Fred
import pandas as pd


def fetch_macro_data(api_key="5363320bd7af2401dafe2914c2c974c8"):

    fred = Fred(api_key)

    interest_rate = fred.get_series("FEDFUNDS")
    inflation_rate = fred.get_series("CPIAUCSL")
    unemployment_rate = fred.get_series("UNRATE")
    gdp_growth = fred.get_series("GDP")

    macro_data = pd.DataFrame(
        {
            "Interest Rate": interest_rate,
            "Inflation Rate": inflation_rate,
            "Unemployment Rate": unemployment_rate,
            "GDP Growth": gdp_growth,
        }
    )

    macro_data.index.name = "Date"

    macro_data.to_csv("./data/us_macro_data.csv", index=True)

    return macro_data
