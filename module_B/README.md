# Module B - 포트폴리오 최적화

# python step1. DCC-GARCH 공분산 추정
import pandas as pd
import numpy as np
import warnings
from arch import arch_model

warnings.filterwarnings('ignore')

print("데이터 불러오는 중...")
base_url = "https://raw.githubusercontent.com/harryk00/stockportfolio26_erica/main/module_B/"
log_returns = pd.read_csv(base_url + "log_returns.csv", index_col=0, parse_dates=True).sort_index().dropna()

def get_dcc_garch_covariance(returns):
    assets = returns.columns
    current_vols = {}
    for asset in assets:
        am = arch_model(returns[asset] * 100, vol='Garch', p=1, q=1, mean='Zero')
        res = am.fit(disp='off')
        pred_var = res.forecast(horizon=1).variance.iloc[-1, 0] / 10000 
        current_vols[asset] = np.sqrt(pred_var)
    
    D = np.diag(list(current_vols.values()))
    R = returns.ewm(span=60).corr().xs(returns.index[-1], level=0).values
    cov_matrix = D @ R @ D
    return pd.DataFrame(cov_matrix, index=assets, columns=assets)

dcc_cov_matrix = get_dcc_garch_covariance(log_returns)

print("\n--- DCC-GARCH 공분산 행렬 계산 완료 ---")
print(dcc_cov_matrix)

pd.set_option('display.max_columns', None)  # 열 생략 없이 전부 다 보여달라는 옵션
pd.set_option('display.width', 1000)        # 화면 너비를 넓게 써서 줄바꿈 안 되게 하는 옵션

print(dcc_cov_matrix)
