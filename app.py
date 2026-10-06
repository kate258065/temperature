import pandas as pd
import numpy as np
import plotly.graph_objects as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="서울 기온 선형회귀 모델 비교 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 선형회귀 모델 구간별 학습 및 평가")

# 데이터 불러오기 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 처리 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df["연도"] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일수 300일 이상 데이터만 필터링
    valid_years = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    
    # 독립변수 X: 1908년 기준 지난 연수 (연도 - 1908)
    valid_years["지난연수"] = valid_years["연도"] - 1908
    return valid_years

df_yearly = load_data()

# ---------------------------------------------------------
# 데이터셋 분할 (테스트 데이터: 2006~2025년)
# ---------------------------------------------------------
test_df = df_yearly[(df_yearly["연도"] >= 2006) & (df_yearly["연도"] <= 2025)].copy()

# 1. 과거 50년 학습 데이터 (1956~2005년)
train_50y_df = df_yearly[(df_yearly["연도"] >= 1956) & (df_yearly["연도"] <= 2005)].copy()

# 2. 과거 100년 학습 데이터 (1906~2005년)
train_100y_df = df_yearly[(df_yearly["연도"] >= 1906) & (df_yearly["연도"] <= 2005)].copy()

# 3. 전체 데이터 (전체 데이터 기반 평가용)
train_full_df = df_yearly.copy()

# ---------------------------------------------------------
# 모델 학습 및 평가 함수
# ---------------------------------------------------------
def train_and_evaluate(train_data, test_data):
    X_train = train_data[["지난연수"]]
    y_train = train_data["연평균기온"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope_per_year = model.coef_[0]
    slope_100y = slope_per_year * 100
    
    X_test = test_data[["지난연수"]]
    y_test = test_data["연평균기온"]
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, slope_100y, mae, mse, r2

# 모델별 학습 및 평가
model_50y, slope_100y_50, mae_50, mse_50, r2_50 = train_and_evaluate(train_50y_df, test_df)
model_100y, slope_100y_100, mae_100, mse_100, r2_100 = train_and_evaluate(train_100y_df, test_df)

# 전체 데이터 모델 (자기 자신 데이터로 평가)
X_full = train_full_df[["지난연수"]]
y_full = train_full_df["연평균기온"]
model_full = LinearRegression()
model_full.fit(X_full, y_full)
y_full_pred = model_full.predict(X_full)

slope_100y_full = model_full.coef_[0] * 100
mae_full = mean_absolute_error(y_full, y_full_pred)
mse_full = mean_squared_error(y_full, y_full_pred)
r2_full = r2_score(y_full, y_full_pred)

# ---------------------------------------------------------
# 상단: 100년당 기울기 및 평가 지표 비교 표
# ---------------------------------------------------------
st.subheader("📊 모델별 100년당 기울기 및 최근 20년(2006~2025) 예측 성능 비교")

# 요약 데이터프레임 생성
summary_data = {
    "모델 구분": [
        "과거 50년 학습 (1956~2005)",
        "과거 100년 학습 (1906~2005)",
        "전체 데이터 (1908~2025)*"
    ],
    "학습 기간": ["1956년 ~ 2005년", "1906년 ~ 2005년", "전체 (1908~2025)"],
    "학습 데이터 수": [len(train_50y_df), len(train_100y_df), len(train_full_df)],
    "100년당 상승량 (°C/100년)": [f"+{slope_100y_50:.2f} °C", f"+{slope_100y_100:.2f} °C", f"+{slope_100y_full:.2f} °C"],
    "테스트 MAE (°C)": [f"{mae_50:.4f}", f"{mae_100:.4f}", f"{mae_full:.4f} (전체)"],
    "테스트 MSE (°C²)": [f"{mse_50:.4f}", f"{mse_100:.4f}", f"{mse_full:.4f} (전체)"],
    "테스트 R²": [f"{r2_50:.4f}", f"{r2_100:.4f}", f"{r2_full:.4f} (전체)"]
}

st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
st.caption("* 전체 데이터 모델은 공통 테스트셋이 아닌 전체 관측 데이터 자체에 대한 평가치입니다.")

st.markdown("---")

# ---------------------------------------------------------
# 핵심 분석 결과 메트릭 카드
# ---------------------------------------------------------
st.subheader("🔍 최근 50년 vs 최근 100년 학습 결과 핵심 비교")

col1, col2, col3 = st.columns(3)

with col1:
    slope_diff = slope_100y_50 - slope_100y_100
    st.metric(
        label="📈 100년당 기온 상승 기울기 차이",
        value=f"+{slope_100y_50:.2f} °C/100년",
        delta=f"100년 학습 대비 +{slope_diff:.2f} °C 더 가파름"
    )

with col2:
    mae_diff = mae_50 - mae_100
    st.metric(
        label="🎯 최근 20년 예측 오차 (MAE)",
        value=f"{mae_50:.3f} °C (50년 학습)",
        delta=f"100년 학습 대비 {abs(mae_diff):.3f} °C 오차 감소" if mae_diff < 0 else f"100년 학습 대비 {mae_diff:.3f} °C 오차 증가",
        delta_color="inverse"
    )

with col3:
    r2_diff = r2_50 - r2_100
    st.metric(
        label="📈 최근 20년 설명력 (R²)",
        value=f"{r2_50:.3f} (50년 학습)",
        delta=f"100년 학습 대비 {r2_diff:+.3f}"
    )

st.markdown("---")

# ---------------------------------------------------------
# 시각화 (Plotly)
# ---------------------------------------------------------
st.subheader("📈 회귀선 및 테스트 데이터(2006~2025) 시각화")

x_range_years = np.arange(1900, 2101)
x_range_passed = x_range_years - 1908

pred_50y = model_50y.predict(x_range_passed.reshape(-1, 1))
pred_100y = model_100y.predict(x_range_passed.reshape(-1, 1))
pred_full = model_full.predict(x_range_passed.reshape(-1, 1))

fig = px.Figure()

# 1. 학습 데이터 (1908~2005)
train_past_df = df_yearly[df_yearly["연도"] <= 2005]
fig.add_trace(px.Scatter(
    x=train_past_df["연도"],
    y=train_past_df["연평균기온"],
    mode="markers",
    name="과거 관측 데이터 (~2005)",
    marker=dict(color="lightslategray", size=6, opacity=0.6)
))

# 2. 공통 테스트 데이터 (2006~2025)
fig.add_trace(px.Scatter(
    x=test_df["연도"],
    y=test_df["연평균기온"],
    mode="markers",
    name="공통 테스트 데이터 (2006~2025)",
    marker=dict(color="crimson", size=9, symbol="diamond")
))

# 3. 과거 50년 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=pred_50y,
    mode="lines",
    name=f"과거 50년 학습 회귀선 (+{slope_100y_50:.2f}°C/100년)",
    line=dict(color="orange", width=2.5)
))

# 4. 과거 100년 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=pred_100y,
    mode="lines",
    name=f"과거 100년 학습 회귀선 (+{slope_100y_100:.2f}°C/100년)",
    line=dict(color="mediumblue", width=2, dash="dash")
))

# 5. 전체 데이터 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=pred_full,
    mode="lines",
    name=f"전체 데이터 회귀선 (+{slope_100y_full:.2f}°C/100년)",
    line=dict(color="green", width=1.5, dash="dot")
))

# 레이아웃 설정
fig.update_layout(
    title="학습 기간별 회귀 직선 및 최근 20년(2006~2025) 예측 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 결과 해석 및 요약
# ---------------------------------------------------------
st.markdown("### 💡 분석 및 비교 결과 요약")
st.markdown(f"""
1. **기울기 변화 (°C/100년)**:
   - **과거 100년(1906~2005) 학습**: 100년당 약 **+{slope_100y_100:.2f} °C** 상승 기울기를 보였습니다.
   - **과거 50년(1956~2005) 학습**: 100년당 약 **+{slope_100y_50:.2f} °C** 상승 기울기로, 최근에 가까운 과거 데이터를 학습할수록 상승 속도가 가파르고 빨라짐을 알 수 있습니다.

2. **최근 20년(2006~2025) 테스트 데이터 예측 성능**:
   - 최근 20년간 서울의 기온 상승 속도가 더욱 가속화되었기 때문에, 상대적으로 최신 경향(가파른 상승 기울기)을 반영한 **과거 50년 학습 모델**이 **과거 100년 학습 모델**보다 최근 20년 기온을 훨씬 더 적은 오차(낮은 MAE, MSE)와 높은 설명력($R^2$)으로 정확하게 예측합니다.
""")
