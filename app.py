import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

df = pd.read_csv(DATA_URL, encoding="utf-8")
df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 연도별 관측일수 및 평균기온
yearly = df.groupby("연도").agg(
    관측일수=("평균기온", "count"),
    연평균기온=("평균기온", "mean")
).reset_index()

# 필터링: 2025년 이하 & 관측일수 300일 이상
filtered = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()

print("전체 이용 가능한 연도:", filtered["연도"].min(), "~", filtered["연도"].max(), f"(총 {len(filtered)}개)")

# 회귀 모델 학습 함수 (1908년 기준 지난 연수 X = 연도 - 1908)
def train_and_eval(train_df, test_df, name):
    X_train = (train_df[["연도"]] - 1908).values
    y_train = train_df["연평균기온"].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    if test_df is not None and len(test_df) > 0:
        X_test = (test_df[["연도"]] - 1908).values
        y_test = test_df["연평균기온"].values
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        # 전체 데이터에 대한 평가 (Self-evaluation / In-sample evaluation)
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
    return {
        "모델명": name,
        "학습 연도": f"{train_df['연도'].min()} ~ {train_df['연도'].max()} ({len(train_df)}개)",
        "기울기(°C/년)": slope,
        "절편": intercept,
        "MAE": mae,
        "MSE": mse,
        "R²": r2
    }

# 1. 전체 데이터 모델
full_model = train_and_eval(filtered, None, "전체 데이터 모델")

# 공통 테스트 데이터: 최근 20년 (2006 ~ 2025)
test_df = filtered[(filtered["연도"] >= 2006) & (filtered["연도"] <= 2025)]

# 2. 최근 50년 학습 모델 (1956 ~ 2005)
train_50_df = filtered[(filtered["연도"] >= 1956) & (filtered["연도"] <= 2005)]
model_50 = train_and_eval(train_50_df, test_df, "최근 50년 학습 (1956~2005)")

# 3. 최근 100년 학습 모델 (1906 ~ 2005)
train_100_df = filtered[(filtered["연도"] >= 1906) & (filtered["연도"] <= 2005)]
model_100 = train_and_eval(train_100_df, test_df, "최근 100년 학습 (1906~2005)")

res_df = pd.DataFrame([full_model, model_100, model_50])
print(res_df.to_string())
