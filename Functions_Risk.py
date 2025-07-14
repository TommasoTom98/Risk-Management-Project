# LIBRARY

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, pearsonr
from matplotlib.colors import LinearSegmentedColormap
import os

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

#######################################################
###################### FUNCTIONS ######################
#######################################################

# Point 1

def Plot_Prices(df, Titles):
    """
    Plot the historical prices for each title in a separate image.
    """
    os.makedirs("img/price", exist_ok=True)
    
    for j in Titles:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="Date", y=j)
        plt.title(f"{j}")
        plt.ylabel("Prices")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"img/price/Prices_{j}.png", dpi=300, bbox_inches='tight')
        plt.show()

def Plot_LogReturns(df, Titles):
    """
    Plot the log returns distributions for each title in a separate image.
    """
    os.makedirs("img/log_returns", exist_ok=True)

    for j in Titles:
        plt.figure(figsize=(12, 6))
        sns.histplot(df, x=f"{j}", bins=100, kde=True)
        plt.title(f"{j}")
        plt.ylabel("Frequency")
        plt.xlabel("Log Returns")
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"img/log_returns/LogReturns_{j}.png", dpi=300, bbox_inches='tight')
        plt.show()


        
# POINT 2

def Returns_of_Titles(df, Titles):
    """
    Compute total return for each title over the full period.
    """
    Titles_Ret = []
    for i in Titles: 
        Tit_clear = df[i].dropna()    
        Titles_Ret.append((Tit_clear[-1] - Tit_clear[0]) / Tit_clear[0])
    Returns_of_Funds = pd.Series(dict(zip(Titles, Titles_Ret)))
    return Returns_of_Funds

def Volatility_Periods(df):
    """
    Compute volatility (std) over daily, monthly, yearly periods.
    """
    Daily_Std = df.std()
    Monthly_Std = df.rolling(window=21,step=21).std().mean()
    Yearly_Std = df.rolling(window=252,step=252).std().mean()
    Monthly_Std_m1 = df.std() * np.sqrt(21)
    Yearly_Std_m1 = df.std() * np.sqrt(252)
    Vol_Periods = pd.DataFrame([Daily_Std, Monthly_Std, Yearly_Std], index=["1D", "1M", "1Y"])
    Vol_Periods_m1 = pd.DataFrame([Daily_Std, Monthly_Std_m1, Yearly_Std_m1], index=["1D", "1M", "1Y"])
    print("Volatility Periods ")
    print(Vol_Periods)
    print("")

def Returns_Periods(df):
    """
    Compute returns over daily, monthly, yearly periods (in %).
    """
    df = df.dropna()
    Daily_return = 100*(df.iloc[-1] - df.iloc[-2])/df.iloc[-2]
    Monthly_return = 100*(df.iloc[-1] - df.iloc[-21]) / df.iloc[-21]
    Yearly_return = 100*(df.iloc[-1] - df.iloc[-252]) / df.iloc[-252]
    Returns_Periods = pd.DataFrame([Daily_return, Monthly_return, Yearly_return], index=["1D", "1M", "1Y"])
    print("Returns Periods (%)")
    print(Returns_Periods)
    print("")


def Moving_Average_of_Titles(df, w):  
    """
    Functions that plot Moving Average of 
    Close prices of titles.
    """
    
    os.makedirs("img/SMA", exist_ok=True)
    
    for col in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[col], label=f'Price {col}', alpha=0.4, color='black')
        for window in w:
            mean_sma = df[col].rolling(window = window, min_periods=1).mean()
            plt.plot(df.index, mean_sma, label=f'SMA {window} days')
        plt.title(f"Simple Moving Averages (SMA) - {col}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"img/SMA/SMA_price{col}.png", dpi=300, bbox_inches='tight')
        plt.show() 
    
def Pearson_Fund_Test(df, Titles):
    """
    Compute p-values for Pearson correlation between all pairs of titles.
    Plots a heatmap of the p-values.
    """
    p_matrix = pd.DataFrame()
    i = 0
    for col in Titles:
        j = 0
        for elem in Titles:    
            valid = df[col].notna() & df[elem].notna()
            _, p_matrix.loc[i, j] = pearsonr(df[col][valid], df[elem][valid])
            j += 1
        i += 1
    fund_names = df.columns
    p_matrix.index = fund_names
    p_matrix.columns = fund_names
    green_red_cmap = LinearSegmentedColormap.from_list("GreenRed", ["green", "red"])
    plt.title("p-values Matrix for the Correlation Indices")
    sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap=green_red_cmap)
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig("img/Pearson_test.png", dpi=300, bbox_inches='tight')

# POINT 3

def VaR(data, alpha):
    """
    Compute Value at Risk (VaR) at given alpha quantile (historical method).
    """
    if isinstance(data, pd.Series):
        result = np.percentile(data.dropna(), 100 * alpha, method='lower')
    else:
        result = data.apply(lambda col: np.percentile(col.dropna(), 100 * alpha, method='lower'))
        if isinstance(result, pd.Series):
            result.index.name = None
    return 100 * result

def gauss_VaR(data, alpha, T):
    """
    Compute Gaussian (parametric) VaR at given alpha and time horizon T.
    """
    if isinstance(data, pd.Series):
        data = data.dropna()
        mu = data.mean()
        std = data.std()
        result = mu * T + norm.ppf(alpha) * std * np.sqrt(T)
    else:
        result = data.apply(lambda col: col.dropna().mean() * T + norm.ppf(alpha) * col.dropna().std() * np.sqrt(T))
        if isinstance(result, pd.Series):
            result.index.name = None
    return 100 * result

def t_VaR(data, alpha):
    """
    Compute VaR using t-student distribution at given alpha.
    """
    if isinstance(data, pd.Series):
        data = data.dropna()
        mu = data.mean()
        std = data.std()
        n = data.shape[0]
        dof = n - 1
        t_crit = t.ppf(alpha, df=dof)
        result = -mu + std * t_crit
    else:
        result = data.apply(lambda col: col.dropna().mean() + col.dropna().std() * t.ppf(alpha, df=col.dropna().shape[0]-1))
        if isinstance(result, pd.Series):
            result.index.name = None
    return 100 * (-result)

def VaR_Methods(df, alpha, T):
    """
    Compute and compare Parametric, Gaussian, and t-student VaR for each fund.
    Returns a DataFrame with all results.
    """
    ParamVaR = VaR(df, alpha).values
    GaussVaR = gauss_VaR(df, alpha, T).values
    delta = 100 * (ParamVaR - GaussVaR) / abs(ParamVaR)
    tVaR = t_VaR(df, alpha).values
    dict = {"Parametric %": ParamVaR,
            "Gaussian %": GaussVaR,
            "Delta %": delta,
            "t-student": tVaR}
    DataF = pd.DataFrame(dict)
    DataF = DataF.set_index(VaR(df, alpha).index)
    DataF.index.name = "VaR"
    return DataF

def VaR_Methods_Portfolio(df, alpha, T):
    """
    Compute Parametric, Gaussian, and t-student VaR for a portfolio (single series).
    Returns a DataFrame with all results.
    """
    ParamVaR = VaR(df, alpha).values
    GaussVaR = gauss_VaR(df, alpha, T).values
    tVaR = t_VaR(df, alpha).values
    dict = {"Parametric": ParamVaR.item(),
            "Gaussian": GaussVaR.item(),
            "t-student": tVaR.item()}
    SeriesP = pd.Series(dict, name="VaR")
    return SeriesP.to_frame().T

def Plot_VaR(df, Titles, alpha):
    """
    Plot the distribution of log returns and overlay VaR and Gaussian VaR for each title.
    """
    n = math.ceil(len(Titles)/2)
    fig, ax = plt.subplots(n, 2, figsize=(20,30))
    for i, j in enumerate(Titles):
        row, col = i // 2, i % 2
        var = VaR(df[j], alpha)/100
        g_var = gauss_VaR(df[j], alpha, 1)/100
        sns.histplot(df, x=f"{j}", ax=ax[row, col], label="Fund's Performance")
        ax[row, col].set_title(f"{j}")
        ax[row, col].axvline(var, c="r", label=rf"VaR($\alpha={alpha}$)")
        ax[row, col].axvline(g_var, c="g", label=rf"Gauss VaR($\alpha={alpha}$)")
        ax[row, col].set_ylabel("Frequency")
        ax[row, col].set_xlabel("Log Returns")
        ax[row, col].tick_params(axis="x", labelrotation=45)
        ax[row, col].legend()
        ax[row, col].grid()
    if len(Titles) % 2 != 0:
        ax[-1, -1].set_visible(False)
    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95)
    plt.tight_layout()
    plt.savefig("img/VaR.png", dpi=500, bbox_inches='tight')
    plt.show()
