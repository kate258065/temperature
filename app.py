import pandas as pd
import numpy as np
import plotly.graph_objects as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측 및 기울기 분석기")

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

# 1. 전체 기간 회귀 모델 (1908년부터 지난 연수를 독립변수로 설정)
df_yearly["지난연수"] = df_yearly["연도"] - 1908

X_full = df_yearly[["지난연수"]]
y_full = df_yearly["연평균기온"]

model_full = LinearRegression()
model_full.fit(X_full, y_full)

# 1년당 기울기 및 100년당 상승폭 (°C/100년)
slope_full_per_year = model_full.coef_[0]
slope_full_100y = slope_full_per_year * 100

# 2. 최근 20년 회귀 모델
max_year = df_yearly["연도"].max()
df_recent20 = df_yearly[df_yearly["연도"] >= (max_year - 19)].copy()

X_recent = df_recent20[["연도"]]
y_recent = df_recent20["연평균기온"]

model_recent = LinearRegression()
model_recent.fit(X_recent, y_recent)

slope_recent_per_year = model_recent.coef_[0]
slope_recent_100y = slope_recent_per_year * 100

# 통계 수치
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())
num_years = len(df_yearly)
corr = np.corrcoef(df_yearly["연도"], df_yearly["연평균기온"])[0, 1]

# ---------------------------------------------------------
# 상단: 100년당 기온 상승량 강조 및 기울기 비교
# ---------------------------------------------------------
st.subheader("🔥 100년당 기온 상승량 비교")

col_slope1, col_slope2, col_slope3 = st.columns(3)

with col_slope1:
    st.metric(
        label=f"📈 전체 기간 100년당 상승량 ({start_year}~{end_year}년)",
        value=f"+{slope_full_100y:.2f} °C",
        delta=f"연간 +{slope_full_per_year:.4f} °C/년"
    )

with col_slope2:
    # 최근 20년과 전체 기간의 속도 차이 계산
    speed_ratio = slope_recent_100y / slope_full_100y if slope_full_100y != 0 else 0
    st.metric(
        label=f"🚀 최근 20년 100년당 상승량 ({end_year-19}~{end_year}년)",
        value=f"+{slope_recent_100y:.2f} °C",
        delta=f"전체 평균 대비 약 {speed_ratio:.1f}배 속도",
        delta_color="normal"
    )

with col_slope3:
    st.metric(
        label="📊 전체 분석 데이터 정보",
        value=f"{num_years}개 해 (상관계수 r = {corr:.4f})",
        help=f"분석 기간: {start_year}년 ~ {end_year}년"
    )

st.markdown("---")

# ---------------------------------------------------------
# 예측 슬라이더 및 시각화
# ---------------------------------------------------------
selected_year = st.slider("예상 기온을 확인하고 싶은 연도를 선택하세요", 1900, 2100, 2026)

# 예측 (전체 기간 모델 기준)
input_years_passed = selected_year - 1908
predicted_temp = model_full.predict([[input_years_passed]])[0]

st.metric(
    label=f"🔮 {selected_year}년 서울 예상 연평균 기온 (전체 기간 회귀선 기준)",
    value=f"{predicted_temp:.2f} °C"
)

# Plotly 그래프 작성
x_range_years = np.arange(start_year, 2101)
x_range_passed = x_range_years - 1908
y_range_full_pred = model_full.predict(x_range_passed.reshape(-1, 1))

# 최근 20년 회귀선의 연장선
y_range_recent_pred = model_recent.predict(x_range_years.reshape(-1, 1))

fig = px.Figure()

# 1. 실제 관측 데이터 (산점도)
fig.add_trace(px.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    marker=dict(color="royalblue", size=7, opacity=0.7)
))

# 2. 전체 기간 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=y_range_full_pred,
    mode="lines",
    name=f"전체 기간 회귀선 (+{slope_full_100y:.2f}°C/100년)",
    line=dict(color="firebrick", width=2.5)
))

# 3. 최근 20년 회귀선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=y_range_recent_pred,
    mode="lines",
    name=f"최근 20년 회귀선 (+{slope_recent_100y:.2f}°C/100년)",
    line=dict(color="orange", width=2, dash="dash")
))

# 4. 슬라이더 선택 연도 강조
fig.add_trace(px.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택 연도({selected_year}년)",
    marker=dict(color="green", size=14, symbol="star")
))

# 레이아웃 설정
fig.update_layout(
    title="서울 연도별 평균기온 추이 및 회귀선 비교 (전체 기간 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)
