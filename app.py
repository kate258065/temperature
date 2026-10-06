import pandas as pd
import numpy as np
import plotly.graph_objects as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기 및 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측 및 모델 평가기")

# 데이터 불러오기 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일수가 300일 이상인 해만 선별
    valid_years = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    return valid_years

df_yearly = load_data()
df_yearly["지난연수"] = df_yearly["연도"] - 1908

# ---------------------------------------------------------
# 1. 모델 데이터셋 분할 및 학습
# ---------------------------------------------------------

# 공통 테스트 데이터 (최근 20년: 2006~2025년)
df_test = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)].copy()
X_test = df_test[["지난연수"]]
y_test = df_test["연평균기온"]

# 훈련 데이터셋 1: 최근 50년 (1956~2005년)
df_train_50 = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)].copy()
X_train_50 = df_train_50[["지난연수"]]
y_train_50 = df_train_50["연평균기온"]

# 훈련 데이터셋 2: 최근 100년 (1906~2005년)
df_train_100 = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)].copy()
X_train_100 = df_train_100[["지난연수"]]
y_train_100 = df_train_100["연평균기온"]

# 모델 학습
model_50 = LinearRegression().fit(X_train_50, y_train_50)
model_100 = LinearRegression().fit(X_train_100, y_train_100)

# 전체 데이터 학습 모델 (기존)
model_full = LinearRegression().fit(df_yearly[["지난연수"]], df_yearly["연평균기온"])

# 최근 20년 자체 학습 모델 (기존)
model_recent20 = LinearRegression().fit(X_test, y_test)

# ---------------------------------------------------------
# 2. 성능 평가 지표 계산 (테스트 데이터 2006~2025 기준)
# ---------------------------------------------------------

def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)
    slope_100y = model.coef_[0] * 100
    return mae, mse, r2, slope_100y

mae_50, mse_50, r2_50, slope_50_100y = evaluate_model(model_50, X_test, y_test)
mae_100, mse_100, r2_100, slope_100_100y = evaluate_model(model_100, X_test, y_test)
mae_full, mse_full, r2_full, slope_full_100y = evaluate_model(model_full, X_test, y_test)

# ---------------------------------------------------------
# 3. 상단 대시보드: 기울기 및 성능 비교 표
# ---------------------------------------------------------
st.subheader("📊 훈련 기간별 모델 기울기 및 최근 20년(2006~2025) 예측 성능 평가")

eval_df = pd.DataFrame({
    "구분": ["과거 50년 학습 모델 (1956~2005)", "과거 100년 학습 모델 (1906~2005)", "전체 기간 학습 모델 (1908~2025)"],
    "학습 데이터 기간": ["1956 ~ 2005년", "1906 ~ 2005년", f"{int(df_yearly['연도'].min())} ~ 2025년"],
    "100년당 기온 상승량 (°C)": [f"+{slope_50_100y:.2f} °C", f"+{slope_100_100y:.2f} °C", f"+{slope_full_100y:.2f} °C"],
    "MAE (평균 절대 오차)": [f"{mae_50:.4f}", f"{mae_100:.4f}", f"{mae_full:.4f}"],
    "MSE (평균 제곱 오차)": [f"{mse_50:.4f}", f"{mse_100:.4f}", f"{mse_full:.4f}"],
    "R² (결정계수)": [f"{r2_50:.4f}", f"{r2_100:.4f}", f"{r2_full:.4f}"]
})

st.table(eval_df)

st.info(
    "💡 **분석 결과 해석:**\n"
    f"- **과거 50년 학습 모델(100년당 +{slope_50_100y:.2f}°C)**은 **과거 100년 학습 모델(100년당 +{slope_100_100y:.2f}°C)**보다 기울기가 더 가파릅니다.\n"
    f"- 최근 20년(2006~2025) 기온이 급격히 상승함에 따라, 최근 과거 50년 데이터로 학습한 모델이 결정계수(R²) 및 오차(MAE/MSE) 측면에서 테스트 데이터를 훨씬 더 정확하게 예측하는 것으로 나타납니다."
)

st.markdown("---")

# ---------------------------------------------------------
# 4. 연도 선택 슬라이더 및 예측값 확인
# ---------------------------------------------------------
selected_year = st.slider("예상 기온을 확인하고 싶은 연도를 선택하세요", 1900, 2100, 2026)
input_years_passed = selected_year - 1908

pred_50 = model_50.predict([[input_years_passed]])[0]
pred_100 = model_100.predict([[input_years_passed]])[0]
pred_full = model_full.predict([[input_years_passed]])[0]

st.write(f"### 🔮 {selected_year}년 모델별 예상 연평균 기온")
col_p1, col_p2, col_p3 = st.columns(3)
col_p1.metric("과거 50년 학습 모델", f"{pred_50:.2f} °C")
col_p2.metric("과거 100년 학습 모델", f"{pred_100:.2f} °C")
col_p3.metric("전체 데이터 학습 모델", f"{pred_full:.2f} °C")

# ---------------------------------------------------------
# 5. Plotly 그래프 시각화
# ---------------------------------------------------------
start_year = int(df_yearly["연도"].min())
x_range_years = np.arange(start_year, 2101)
x_range_passed = x_range_years - 1908

y_range_50 = model_50.predict(x_range_passed.reshape(-1, 1))
y_range_100 = model_100.predict(x_range_passed.reshape(-1, 1))
y_range_full = model_full.predict(x_range_passed.reshape(-1, 1))

fig = px.Figure()

# 훈련 데이터 (과거)
fig.add_trace(px.Scatter(
    x=df_yearly[df_yearly["연도"] < 2006]["연도"],
    y=df_yearly[df_yearly["연도"] < 2006]["연평균기온"],
    mode="markers",
    name="과거 관측 데이터 (~2005)",
    marker=dict(color="steelblue", size=7, opacity=0.6)
))

# 테스트 데이터 (최근 20년)
fig.add_trace(px.Scatter(
    x=df_test["연도"],
    y=df_test["연평균기온"],
    mode="markers",
    name="테스트 데이터 (2006~2025)",
    marker=dict(color="red", size=9, symbol="circle")
))

# 회귀선들
fig.add_trace(px.Scatter(
    x=x_range_years, y=y_range_50,
    mode="lines", name=f"50년 학습 회귀선 (+{slope_50_100y:.2f}°C/100년)",
    line=dict(color="orange", width=2.5, dash="dash")
))

fig.add_trace(px.Scatter(
    x=x_range_years, y=y_range_100,
    mode="lines", name=f"100년 학습 회귀선 (+{slope_100_100y:.2f}°C/100년)",
    line=dict(color="purple", width=2.5, dash="dot")
))

fig.add_trace(px.Scatter(
    x=x_range_years, y=y_range_full,
    mode="lines", name=f"전체 학습 회귀선 (+{slope_full_100y:.2f}°C/100년)",
    line=dict(color="green", width=2)
))

# 선택 연도 강조
fig.add_trace(px.Scatter(
    x=[selected_year, selected_year, selected_year],
    y=[pred_50, pred_100, pred_full],
    mode="markers",
    name=f"선택 연도({selected_year}년) 예측점",
    marker=dict(color="black", size=11, symbol="star")
))

fig.update_layout(
    title="서울 연평균 기온 회귀 모델 예측 성능 및 추이 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
