import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go


# 모델 로드
model_base = joblib.load("Models/model_base_mid.pkl")
model_lag = joblib.load("Models/model_lag_mid.pkl")

# feature 순서 로드
feature_columns_base = joblib.load(
    "Features/feature_columns_base.pkl"
)
feature_columns_lag = joblib.load(
    "Features/feature_columns_lag.pkl"
)

st.title("훈련소 인플루엔자 위험 예측")

# 사용자 입력
avg_temp = st.slider(
    "평균기온",
    -20.0,
    40.0,
    0.0,
    step=0.1
)

avg_1_temp = st.slider(
    "1주 전 평균기온",
    -20.0,
    40.0,
    0.0,
    step=0.1
)

avg_2_temp = st.slider(
    "2주 전 평균기온",
    -20.0,
    40.0,
    0.0,
    step=0.1
)

min_temp = st.slider(
    "최저기온",
    -20.0,
    40.0,
    0.0,
    step=0.1
)

avg_wind = st.slider(
    "평균 풍속",
    0.0,
    10.0,
    2.0,
    step=0.1
)
avg_humidity = st.slider(
    "평균 습도",
    0.0,
    100.0,
    50.0,
    step=0.1
)

temp_diff=avg_temp - min_temp

pm10 = st.slider(
    "PM10",
    0.0,
    200.0,
    30.0,
    step=1.0
)

prev_ili = st.number_input(
    "전주 ILI",
    0.0,
    100.0,
    10.0,
    help="전 주 인플루엔자 ILI 값입니다. 질병관리청: https://dportal.kdca.go.kr/pot/is/st/influ.do 에서 참고 가능합니다."
)

prev_1_ili = st.number_input(
    "전전주 ILI",
    0.0,
    100.0,
    10.0,
    help="전전 주 인플루엔자 ILI 값입니다. 질병관리청: https://dportal.kdca.go.kr/pot/is/st/influ.do 에서 참고 가능합니다."
)

temp_ma_3 = np.mean([
    avg_temp,
    avg_1_temp,
    avg_2_temp
])

# week encoding 예시
week = 10

week_sin = np.sin(
    2 * np.pi * week / 52
)

week_cos = np.cos(
    2 * np.pi * week / 52
)




# dataframe 생성
input_df_base = pd.DataFrame([{
    "week": week,
    "avg_temp": avg_temp,
    "min_temp": min_temp,
    "avg_wind": avg_wind,
    "avg_humidity": avg_humidity,
    "temp_diff":temp_diff,
    "week_sin": week_sin,
    "week_cos": week_cos,
    "temp_ma_3":temp_ma_3,
    "미세먼지": pm10
    
}])
input_df_lag = pd.DataFrame([{
    "week": week,
    "avg_temp": avg_temp,
    "min_temp": min_temp,
    "avg_wind": avg_wind,
    "avg_humidity": avg_humidity,
    "temp_diff":temp_diff,
    "week_sin": week_sin,
    "week_cos": week_cos,
    "temp_ma_3":temp_ma_3,
    "미세먼지": pm10,
    "ili_lag_1": prev_ili,
    "ili_lag_2": prev_1_ili
    
}])

# feature 순서 맞춤
input_df_base = input_df_base[feature_columns_base]
input_df_lag = input_df_lag[feature_columns_lag]

# 예측
pred_ili_base = model_base.predict(input_df_base)[0]
pred_ili_lag = model_lag.predict(input_df_lag)[0]

st.metric(
    "예측 ILI without Previous ILI",
    round(pred_ili_base, 2)
)

st.metric(
    "예측 ILI with Previous ILI",
    round(pred_ili_lag, 2)
)

n=st.number_input(
    "훈련병 입영 수",
    0,
    300,
    200,
    help="이번 기수 입영하는 훈련병의 수입니다."
)

#확률
p=pred_ili_lag/1000
prob_at_least_one = 1 - (1-p)**200

st.metric(
    "인플루엔자가 단 한명도 발생하지 않을 확률",
    round(prob_at_least_one, 2)
)

#초기 상태
I0 = np.random.binomial(200, p)
E0 = 0
S0 = n - I0
R0 = 0

#파라미터
Beta=st.slider(
    "감염률",
    0.0,
    1.0,
    0.3,
    step=0.3
)
sigma = 1/2
gamma = 1/5

# 초기 벡터
y0 = S0, E0, I0, R0

# 시간축
t = np.linspace(0, 30, 31)
def deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt

# 적분
ret = odeint(
    deriv,
    y0,
    t,
    args=(n, Beta, sigma, gamma)
)

S, E, I, R = ret.T

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=E,
        mode='lines',
        name='잠복군'
    )
)

fig.add_trace(
    go.Scatter(
        x=t,
        y=I,
        mode='lines',
        name='감염군'
    )
)

fig.update_layout(
    title="SEIR Simulation",
    xaxis_title="Days",
    yaxis_title="People"
)

st.plotly_chart(fig)