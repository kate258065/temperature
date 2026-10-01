import pandas as pd
import numpy as np
import plotly.graph_objects as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

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

# 회귀 모델 학습 (1908년부터 지난 연수를 독립변수 X로 지정)
df_yearly["지난연수"] = df_yearly["연도"] - 1908

X = df_yearly[["지난연수"]]
y = df_yearly["연평균기온"]

model = LinearRegression()
model.fit(X, y)

# 통계 수치 계산
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())
num_years = len(df_yearly)
corr = np.corrcoef(df_yearly["연도"], df_yearly["연평균기온"])[0, 1]

# 정보 표시 (요약 메트릭)
st.subheader("📊 학습 데이터 정보 및 상관계수")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("분석 연도 개수", f"{num_years}개 해")
col_info2.metric("시작 연도", f"{start_year}년")
col_info3.metric("끝 연도", f"{end_year}년")
col_info4.metric("상관계수 (r)", f"{corr:.4f}")

st.markdown("---")

# 연도 선택 슬라이더 (1900년 ~ 2100년)
selected_year = st.slider("예상 기온을 확인하고 싶은 연도를 선택하세요", 1900, 2100, 2026)

# 선택한 연도의 기온 예측
input_years_passed = selected_year - 1908
predicted_temp = model.predict([[input_years_passed]])[0]

# 예측 결과 강조 표시
st.metric(
    label=f"🔮 {selected_year}년 서울 예상 연평균 기온",
    value=f"{predicted_temp:.2f} °C"
)

# Plotly 시각화
# 1908년부터 2100년까지의 회귀선 데이터 생성
x_range_years = np.arange(start_year, 2101)
x_range_passed = x_range_years - 1908
y_range_pred = model.predict(x_range_passed.reshape(-1, 1))

fig = px.Figure()

# 1. 실제 관측 데이터 (산점도)
fig.add_trace(px.Scatter(
    x=df_yearly["연도"],
    y=df_yearly["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    marker=dict(color="royalblue", size=7)
))

# 2. 회귀 직선
fig.add_trace(px.Scatter(
    x=x_range_years,
    y=y_range_pred,
    mode="lines",
    name="회귀 직선",
    line=dict(color="firebrick", width=2)
))

# 3. 슬라이더로 선택한 연도 강조 표시
fig.add_trace(px.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택 연도({selected_year}년)",
    marker=dict(color="orange", size=14, symbol="star")
))

# 레이아웃 설정
fig.update_layout(
    title="서울 연도별 평균기온 추이 및 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
