import pandas as pd
import numpy as np
import plotly.graph_objects as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="서울 기온 회귀 모델 평가 및 비교", layout="wide")
st.title("🌡️ 서울 연평균 기온 회귀 모델 훈련/테스트 성능 비교")

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
    
    # 지난 연수 (독립변수 X): 기준 연도 1908년
    valid_years["지난연수"] = valid_years["연도"] - 1908
    return valid_years

df_yearly = load_data()

# ---------------------------------------------------------
# 데이터 분할 (공통 테스트 데이터: 최근 20년 2006 ~ 2025년)
# ---------------------------------------------------------
test_df = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)]
X_test = test_df[["지난연수"]]
y_test = test_df["연평균기온"]

# 1. 최근 100년 학습 데이터 (1906 ~ 2005년)
train_100_df = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)]
X_train_100 = train_100_df[["지난연수"]]
y_train_100 = train_100_df["연평균기온"]

# 2. 최근 50년 학습 데이터 (1956 ~ 2005년)
train_50_df = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)]
X_train_50 = train_50_df[["지난연수"]]
y_train_50 = train_50_df["연평균기온"]

# ---------------------------------------------------------
# 모델 학습 및 테스트 데이터 평가
# ---------------------------------------------------------
# 모델 1: 최근 100년 (1906~2005)
model_100 = LinearRegression()
model_100.fit(X_train_100, y_train_100)
pred_100_test = model_100.predict(X_test)

slope_100_100y = model_100.coef_[0] * 100
mae_100 = mean_absolute_error(y_test, pred_100_test)
mse_100 = mean_squared_error(y_test, pred_100_test)
r2_100 = r2_score(y_test, pred_100_test)

# 모델 2: 최근 50년 (1956~2005)
model_50 = LinearRegression()
model_50.fit(X_train_50, y_train_50)
pred_50_test = model_50.predict(X_test)

slope_50_100y = model_50.coef_[0] * 100
mae_50 = mean_absolute_error(y_test, pred_50_test)
mse_50 = mean_squared_error(y_test, pred_50_test)
r2_50 = r2_score(y_test, pred_50_test)

# ---------------------------------------------------------
# 화면 출력 및 성능 비교
# ---------------------------------------------------------
st.subheader("📋 모델별 기울기 및 테스트 데이터(2006~2025) 성능 평가")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔹 최근 100년 학습 모델 (1906~2005)")
    st.write(f"- **학습 데이터 수:** {len(train_100_df)}개 해")
    st.write(f"- **기울기 (100년당 상승량):** `+{slope_100_100y:.2f} °C`")
    st.metric("MAE (평균 절대 오차)", f"{mae_100:.4f} °C")
    st.metric("MSE (평균 제곱 오차)", f"{mse_100:.4f}")
    st.metric("R² (결정계수)", f"{r2_100:.4f}")

with col2:
    st.markdown("### 🔸 최근 50년 학습 모델 (1956~2005)")
    st.write(f"- **학습 데이터 수:** {len(train_50_df)}개 해")
    st.write(f"- **기울기 (100년당 상승량):** `+{slope_50_100y:.2f} °C`")
    st.metric("MAE (평균 절대 오차)", f"{mae_50:.4f} °C")
    st.metric("MSE (평균 제곱 오차)", f"{mse_50:.4f}")
    st.metric("R² (결정계수)", f"{r2_50:.4f}")

st.markdown("---")

# ---------------------------------------------------------
# 비교 요약 표
# ---------------------------------------------------------
st.subheader("📊 성능 비교 요약")
summary_df = pd.DataFrame({
    "구분": ["최근 100년 학습 (1906~2005)", "최근 50년 학습 (1956~2005)"],
    "학습 기간": ["1906년 ~ 2005년", "1956년 ~ 2005년"],
    "100년당 기울기 (°C)": [f"+{slope_100_100y:.2f}", f"+{slope_50_100y:.2f}"],
    "MAE (°C)": [f"{mae_100:.4f}", f"{mae_50:.4f}"],
    "MSE": [f"{mse_100:.4f}", f"{mse_50:.4f}"],
    "R²": [f"{r2_100:.4f}", f"{r2_50:.4f}"]
})
st.table(summary_df)

st.markdown("---")

# ---------------------------------------------------------
# Plotly 시각화 (학습 구간, 테스트 구간, 두 회귀선)
# ---------------------------------------------------------
st.subheader("📈 회귀선 및 테스트 구간 비교 시각화")

x_range_years = np.arange(1900, 2026)
x_range_passed = x_range_years - 1908

y_range_100_pred = model_100.predict(x_range_passed.reshape(-1, 1))
y_range_50_pred = model_50.predict(x_range_passed.reshape(-1, 1))

fig = px.Figure()

# 1. 학습 데이터 (1906~2005)
fig.add_trace(px.Scatter(
    x=df_yearly[df_yearly["연도"] <= 2005]["연도"],
    y=df_yearly[df_yearly["연도"] <= 2005]["연평균기온"],
    mode="markers",
    name="과거 관측 데이터 (~2005)",
    marker=dict(color="steelblue", size=6, opacity=0.6)
))

# 2. 테스트 데이터 (2006~2025)
fig.add_trace(px.Scatter(
    x=test_df["연도"],
    y=test_df["연평균기온"],
    mode="markers",
    name="테스트 데이터 (2006~2025)",
    marker=dict(color="crimson", size=9, symbol="diamond")
))

# 3. 최근 100년 학습 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=y_range_100_pred,
    mode="lines",
    name=f"100년 학습 회귀선 (+{slope_100_100y:.2f}°C/100년)",
    line=dict(color="blue", width=2, dash="dash")
))

# 4. 최근 50년 학습 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=y_range_50_pred,
    mode="lines",
    name=f"50년 학습 회귀선 (+{slope_50_100y:.2f}°C/100년)",
    line=dict(color="darkorange", width=2.5)
))

# 레이아웃 설정
fig.update_layout(
    title="서울 기온 회귀 모델 비교 (테스트 구간: 2006년 ~ 2025년)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
