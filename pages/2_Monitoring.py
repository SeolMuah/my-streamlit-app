import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 캐시된 데이터 로더 import
from util.data_loader import (
    load_sensor_data, 
    load_labels, 
    load_models,
    extract_features_from_window,
    detect_anomaly,
    SENSOR_FREQUENCIES
)

st.set_page_config(page_title="실시간 모니터링", page_icon="🔍", layout="wide")

# CSS 스타일 최적화
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)



def get_component_threshold(comp_name, method='svm_default'):
    """
    부품별 임계값 반환
    
    Args:
        comp_name: 부품명
        method: 'svm_default' (SVM 기본값 0) 또는 'normalized' (정규화 후 0.5)
    """
    if method == 'svm_default':
        # SVM 기본 임계값 (decision_function = 0)
        return 0.0
    elif method == 'normalized':
        # 정규화 후 임계값
        thresholds = {
            'cooler': 0.5,
            'valve': 0.5,
            'pump': 0.5,
            'hydraulic': 0.5
        }
        return thresholds.get(comp_name, 0.5)
    else:
        return 0.0

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if 'monitoring_initialized' not in st.session_state:
        st.session_state.monitoring_initialized = False
        st.session_state.monitoring_active = False
        st.session_state.data_ready = False
        st.session_state.models_ready = False
        st.session_state.current_cycle = 0
        st.session_state.current_window = 0
        st.session_state.total_windows = 0
        st.session_state.start_time = None
        st.session_state.anomaly_counts = {}
        st.session_state.data_buffer = {
            'time': deque(maxlen=50),
            'anomaly_scores': {},
            'anomaly_flags': {},
            'sensor_values': {},
            'raw_scores': {},
            'normalized_scores': {}
        }
        st.session_state.alert_history = deque(maxlen=100)
        st.session_state.score_method = 'svm_default'  # 기본값은 SVM 원본 점수
        st.session_state.monitoring_initialized = True

def check_cached_data():
    """캐시된 데이터 확인 및 준비"""
    try:
        # 캐시된 데이터 로드 시도
        sensor_data = load_sensor_data(show_progress=False)
        labels = load_labels(show_progress=False)
        
        if sensor_data and labels is not None:
            # 데이터 버퍼 초기화
            st.session_state.sensor_data = sensor_data
            st.session_state.labels = labels
            st.session_state.total_cycles = len(labels)
            st.session_state.data_ready = True
            
            # 센서 데이터 버퍼 초기화
            for sensor in SENSOR_FREQUENCIES.keys():
                if sensor in sensor_data:
                    st.session_state.data_buffer['sensor_values'][sensor] = deque(maxlen=50)
            
            return True, len(sensor_data)
        else:
            return False, "캐시된 데이터를 찾을 수 없습니다."
    except Exception as e:
        return False, str(e)

def check_cached_models():
    """캐시된 모델 확인 및 준비"""
    try:
        # 캐시된 모델 로드 시도
        models, scalers, metadata = load_models(show_progress=False)
        
        if models is not None:
            st.session_state.models = models
            st.session_state.scalers = scalers
            st.session_state.metadata = metadata or {}
            st.session_state.models_ready = True
            
            # 부품별 데이터 버퍼 초기화
            for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
                st.session_state.data_buffer['anomaly_scores'][comp_name] = deque(maxlen=50)
                st.session_state.data_buffer['anomaly_flags'][comp_name] = deque(maxlen=50)
                st.session_state.data_buffer['raw_scores'][comp_name] = deque(maxlen=50)
                st.session_state.data_buffer['normalized_scores'][comp_name] = deque(maxlen=50)
                st.session_state.anomaly_counts[comp_name] = 0
            
            return True, len(models)
        else:
            return False, "캐시된 모델을 찾을 수 없습니다."
    except Exception as e:
        return False, str(e)

def get_window_data():
    """현재 윈도우 데이터 가져오기"""
    if not st.session_state.data_ready:
        return None
    
    try:
        window_data = {}
        window_size = 10  # 10초 윈도우
        windows_per_cycle = 6  # 60초 / 10초 = 6개 윈도우
        
        current_cycle = st.session_state.current_cycle
        current_window = st.session_state.current_window
        
        # 사이클 끝에 도달하면 다음 사이클로
        if current_cycle >= st.session_state.total_cycles:
            st.session_state.current_cycle = 0
            st.session_state.current_window = 0
            current_cycle = 0
            current_window = 0
        
        # 각 센서에서 데이터 추출
        for sensor_name, sensor_df in st.session_state.sensor_data.items():
            # 센서별 샘플링 레이트
            rate = SENSOR_FREQUENCIES.get(sensor_name, 1)
            
            # 윈도우 내 샘플 수
            samples_per_window = rate * window_size
            
            # 현재 사이클 데이터
            if current_cycle >= len(sensor_df):
                continue
            
            cycle_data = sensor_df.iloc[current_cycle].values
            
            # 윈도우 데이터 추출
            start_idx = current_window * samples_per_window
            end_idx = min(start_idx + samples_per_window, len(cycle_data))
            
            if start_idx < len(cycle_data):
                window_data[sensor_name] = cycle_data[start_idx:end_idx]
        
        # 다음 윈도우로 이동
        st.session_state.current_window += 1
        if st.session_state.current_window >= windows_per_cycle:
            st.session_state.current_window = 0
            st.session_state.current_cycle += 1
        
        return window_data
    except Exception as e:
        st.error(f"데이터 가져오기 중 오류: {e}")
        return None

def process_window(window_data):
    """윈도우 데이터 처리"""
    if not window_data:
        return {}
    
    try:
        # 특징 추출 (캐시된 함수 사용)
        features = extract_features_from_window(window_data)
        
        # 현재 시간 기록
        current_time = datetime.now()
        st.session_state.data_buffer['time'].append(current_time)
        st.session_state.total_windows += 1
        
        # 각 부품 예측 (캐시된 함수 사용)
        anomaly_results = detect_anomaly(features, st.session_state.models, 
                                       st.session_state.scalers, st.session_state.metadata)
        
        # 결과 저장
        processed_results = {}
        score_method = st.session_state.score_method
        
        for comp_name, result in anomaly_results.items():
            raw_score = result['score']  # decision_function의 raw score
            
            # 점수 처리 방식에 따라 분기
            if score_method == 'svm_default':
                # SVM 기본 방식: raw score와 임계값 0 사용
                display_score = raw_score
                threshold = get_component_threshold(comp_name, 'svm_default')
                is_anomaly = raw_score > threshold  # 양수이면 이상
            else:
                # 정규화 방식: 점수를 0-1로 변환
                normalized_score = normalize_decision_score(raw_score, 'sigmoid')
                display_score = normalized_score
                threshold = get_component_threshold(comp_name, 'normalized')
                is_anomaly = normalized_score > threshold
            
            # 결과 저장
            st.session_state.data_buffer['raw_scores'][comp_name].append(raw_score)
            
            if score_method == 'svm_default':
                st.session_state.data_buffer['anomaly_scores'][comp_name].append(raw_score)
            else:
                normalized_score = normalize_decision_score(raw_score, 'sigmoid')
                st.session_state.data_buffer['anomaly_scores'][comp_name].append(normalized_score)
                st.session_state.data_buffer['normalized_scores'][comp_name].append(normalized_score)
            
            st.session_state.data_buffer['anomaly_flags'][comp_name].append(1 if is_anomaly else 0)
            
            if is_anomaly:
                st.session_state.anomaly_counts[comp_name] += 1
            
            # 알림 히스토리 추가
            comp_name_kr = comp_name.replace('cooler', '냉각기').replace('valve', '밸브').replace('pump', '펌프').replace('hydraulic', '유압시스템')
            alert = {
                'time': current_time.strftime('%H:%M:%S'),
                'component': comp_name_kr,
                'is_anomaly': is_anomaly,
                'raw_score': raw_score,
                'display_score': display_score,
                'threshold': threshold,
                'score_method': score_method,
                'model': 'OneClassSVM'
            }
            st.session_state.alert_history.appendleft(alert)
            
            # 처리된 결과 저장
            processed_results[comp_name] = {
                'is_anomaly': is_anomaly,
                'raw_score': raw_score,
                'display_score': display_score,
                'threshold': threshold
            }
        
        # 센서 값 저장
        for sensor_name, sensor_values in window_data.items():
            if sensor_name in st.session_state.data_buffer['sensor_values']:
                st.session_state.data_buffer['sensor_values'][sensor_name].append(
                    np.mean(sensor_values)
                )
        
        return processed_results
        
    except Exception as e:
        st.error(f"데이터 처리 중 오류: {e}")
        return {}

def create_monitoring_plots():
    """모니터링 그래프 생성"""
    if st.session_state.total_windows == 0:
        return None
    
    # 시간 데이터
    times = list(st.session_state.data_buffer['time'])
    time_strings = [t.strftime('%H:%M:%S') for t in times]
    
    # 부품별 이상 탐지 그래프
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['냉각기', '밸브', '펌프', '유압시스템'],
        vertical_spacing=0.30,  # 세로 간격 더 증가
        horizontal_spacing=0.15
    )
    
    # 서브플롯 타이틀 스타일 개선
    fig.update_annotations(
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        borderpad=4
    )
    
    comp_names = ['cooler', 'valve', 'pump', 'hydraulic']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA502']
    
    score_method = st.session_state.score_method
    
    for idx, comp_name in enumerate(comp_names):
        if comp_name in st.session_state.data_buffer['anomaly_scores']:
            row, col = positions[idx]
            
            # 점수 데이터
            display_scores = list(st.session_state.data_buffer['anomaly_scores'][comp_name])
            raw_scores = list(st.session_state.data_buffer['raw_scores'][comp_name])
            flags = list(st.session_state.data_buffer['anomaly_flags'][comp_name])
            threshold = get_component_threshold(comp_name, 'svm_default' if score_method == 'svm_default' else 'normalized')
            
            # 점수 라인
            score_name = "Raw Score" if score_method == 'svm_default' else "Normalized Score"
            comp_name_kr = comp_name.replace('cooler', '냉각기').replace('valve', '밸브').replace('pump', '펌프').replace('hydraulic', '유압시스템')
            
            fig.add_trace(
                go.Scatter(
                    x=time_strings, 
                    y=display_scores,
                    mode='lines+markers',
                    name=f'{comp_name_kr}',  # 한글 이름 사용
                    line=dict(color=colors[idx], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'{comp_name_kr}<br>시간: %{{x}}<br>{score_name}: %{{y:.3f}}<br>Raw Score: %{{customdata:.3f}}<extra></extra>',
                    customdata=raw_scores,
                    legendgroup=f'group{idx}',  # 그룹으로 묶기
                    showlegend=True
                ),
                row=row, col=col
            )
            
            # 이상 탐지 마커
            anomaly_times = [time_strings[i] for i, flag in enumerate(flags) if flag == 1]
            anomaly_scores = [display_scores[i] for i, flag in enumerate(flags) if flag == 1]
            anomaly_raw_scores = [raw_scores[i] for i, flag in enumerate(flags) if flag == 1]
            
            if anomaly_times:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_times, 
                        y=anomaly_scores,
                        mode='markers',
                        name='이상 탐지',
                        marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
                        showlegend=(idx == 0),  # 첫 번째 플롯에서만 범례 표시
                        hovertemplate='🚨 이상 탐지<br>시간: %{x}<br>점수: %{y:.3f}<br>Raw: %{customdata:.3f}<extra></extra>',
                        customdata=anomaly_raw_scores,
                        legendgroup='anomaly'
                    ),
                    row=row, col=col
                )
            
            # 임계값 라인
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color="gray",
                line_width=1,
                annotation_text=f"임계값",
                annotation_position="left",
                annotation_font_size=10,
                row=row, col=col
            )
            
            # Y축 설정 - 동적 범위 계산
            if score_method == 'svm_default':
                # Raw score의 동적 범위 계산
                all_scores = list(st.session_state.data_buffer['anomaly_scores'][comp_name])
                
                if len(all_scores) > 0:
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    
                    # 임계값 0이 항상 포함되도록 보장
                    min_score = min(min_score, -0.5)
                    max_score = max(max_score, 0.5)
                    
                    # 여유 공간 추가 (10%)
                    range_size = max_score - min_score
                    margin = range_size * 0.1
                    
                    y_min = min_score - margin
                    y_max = max_score + margin
                else:
                    y_min, y_max = -2, 2
                
                fig.update_yaxes(
                    title_text="Decision Score", 
                    range=[y_min, y_max], 
                    row=row, col=col
                )
            else:
                # Normalized score 범위
                fig.update_yaxes(
                    title_text="Normalized Score", 
                    range=[0, 1], 
                    row=row, col=col
                )
            
            # X축 설정
            fig.update_xaxes(
                title_text="시간",
                tickangle=45,
                row=row, col=col
            )
    
    # 전체 레이아웃 설정
    title = "실시간 이상 탐지 모니터링"
    if score_method == 'svm_default':
        title += " (SVM Raw Score)"
    else:
        title += " (Normalized Score)"
    
    fig.update_layout(
        height=700,  # 높이 더 증가
        title={
            'text': title,
            'x': 0.5,  # 중앙 정렬
            'xanchor': 'center',
            'font': {'size': 16}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,  # 범례를 더 아래로 배치
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(
            l=50,
            r=50,
            t=80,  # 위쪽 여백
            b=150  # 아래쪽 여백 더 증가 (범례 공간 확보)
        ),
        hovermode='closest'
    )
    
    return fig

def create_status_metrics():
    """상태 메트릭 생성"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_anomalies = sum(st.session_state.anomaly_counts.values())
        st.metric("총 이상 탐지", total_anomalies)
    
    with col2:
        st.metric("처리된 윈도우", st.session_state.total_windows)
    
    with col3:
        current_cycle = st.session_state.current_cycle + 1
        total_cycles = st.session_state.total_cycles
        st.metric("현재 사이클", f"{current_cycle}/{total_cycles}")
    
    with col4:
        if st.session_state.start_time:
            elapsed = datetime.now() - st.session_state.start_time
            elapsed_str = str(elapsed).split('.')[0]
            st.metric("모니터링 시간", elapsed_str)
        else:
            st.metric("모니터링 시간", "00:00:00")
    
    # 현재 이상 부품 수 계산 (추가 행)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 최근 알림에서 이상 상태 확인
        current_anomalies = 0
        if st.session_state.alert_history:
            recent_alerts = list(st.session_state.alert_history)[:4]  # 최근 4개 (부품별 1개씩)
            comp_status = {}
            for alert in recent_alerts:
                comp = alert['component']
                if comp not in comp_status:
                    comp_status[comp] = alert['is_anomaly']
            current_anomalies = sum(comp_status.values())
        
        st.metric("현재 이상 부품", current_anomalies)
    
    with col2:
        # 전체 이상률
        total_possible = st.session_state.total_windows * 4  # 4개 부품
        anomaly_rate = (total_anomalies / total_possible * 100) if total_possible > 0 else 0
        st.metric("전체 이상률", f"{anomaly_rate:.1f}%")
    
    with col3:
        # 가장 문제가 많은 부품
        if st.session_state.anomaly_counts:
            max_comp = max(st.session_state.anomaly_counts.items(), key=lambda x: x[1])
            comp_kr = max_comp[0].replace('cooler', '냉각기').replace('valve', '밸브').replace('pump', '펌프').replace('hydraulic', '유압시스템')
            st.metric("최다 이상 부품", comp_kr, delta=f"{max_comp[1]}건")
    
    with col4:
        # 평균 점수
        all_scores = []
        for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
            if comp_name in st.session_state.data_buffer['anomaly_scores']:
                scores = list(st.session_state.data_buffer['anomaly_scores'][comp_name])
                all_scores.extend(scores)
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            score_label = "평균 점수 (Raw)" if st.session_state.score_method == 'svm_default' else "평균 점수 (Norm)"
            st.metric(score_label, f"{avg_score:.3f}")
        else:
            st.metric("평균 점수", "0.000")

def main():
    """메인 함수"""
    st.title("🔍 실시간 모니터링")
    st.markdown("---")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 기본 업데이트 간격 설정
    update_interval = 2
    
    # 페이지 로드 시 캐시된 데이터 자동 확인
    if not st.session_state.data_ready:
        data_success, data_result = check_cached_data()
        if data_success:
            st.session_state.data_ready = True
    
    if not st.session_state.models_ready:
        model_success, model_result = check_cached_models()
        if model_success:
            st.session_state.models_ready = True
    
    # 사이드바 - 상태 표시 및 제어
    with st.sidebar:
        st.header("📊 시스템 상태")
        
        # 데이터 상태
        if st.session_state.data_ready:
            st.success("✅ 데이터 준비됨")
            st.metric("사이클 수", st.session_state.total_cycles)
            st.metric("센서 수", len(st.session_state.sensor_data))
        else:
            st.error("❌ 데이터 없음")
        
        # 모델 상태
        if st.session_state.models_ready:
            st.success("✅ 모델 준비됨")
            st.metric("모델 수", len(st.session_state.models))
        else:
            st.error("❌ 모델 없음")
        
        # 모니터링 상태
        if st.session_state.monitoring_active:
            st.success("🔴 모니터링 중")
        else:
            st.info("⚪ 대기 중")
        
        st.divider()
        
        # 제어 버튼들
        st.subheader("🎮 제어 패널")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 시작", 
                        disabled=st.session_state.monitoring_active or not (st.session_state.data_ready and st.session_state.models_ready),
                        help="모니터링 시작",
                        use_container_width=True):
                st.session_state.monitoring_active = True
                st.session_state.start_time = datetime.now()
                st.rerun()
        
        with col2:
            if st.button("⏸️ 정지", 
                        disabled=not st.session_state.monitoring_active,
                        help="모니터링 정지",
                        use_container_width=True):
                st.session_state.monitoring_active = False
                st.rerun()
        
        # 추가 제어 버튼들
        if st.button("🔄 새로고침", 
                    help="캐시된 데이터와 모델을 새로고침",
                    use_container_width=True):
            st.session_state.data_ready = False
            st.session_state.models_ready = False
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("🗑️ 초기화", 
                    help="모니터링 데이터를 초기화하고 처음부터 시작",
                    use_container_width=True):
            st.session_state.data_buffer = {
                'time': deque(maxlen=50),
                'anomaly_scores': {},
                'anomaly_flags': {},
                'sensor_values': {},
                'raw_scores': {},
                'normalized_scores': {}
            }
            st.session_state.total_windows = 0
            st.session_state.current_cycle = 0
            st.session_state.current_window = 0
            st.session_state.anomaly_counts = {}
            st.session_state.alert_history = deque(maxlen=100)
            st.session_state.start_time = None
            
            # 버퍼 재초기화
            if st.session_state.data_ready and st.session_state.models_ready:
                for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
                    st.session_state.data_buffer['anomaly_scores'][comp_name] = deque(maxlen=50)
                    st.session_state.data_buffer['anomaly_flags'][comp_name] = deque(maxlen=50)
                    st.session_state.data_buffer['raw_scores'][comp_name] = deque(maxlen=50)
                    st.session_state.data_buffer['normalized_scores'][comp_name] = deque(maxlen=50)
                    st.session_state.anomaly_counts[comp_name] = 0
                
                for sensor in SENSOR_FREQUENCIES.keys():
                    if sensor in st.session_state.sensor_data:
                        st.session_state.data_buffer['sensor_values'][sensor] = deque(maxlen=50)
            
            st.rerun()
        
        # 설정
        st.divider()
        st.subheader("⚙️ 설정")
        
        # 점수 처리 방식 선택
        score_method = st.selectbox(
            "점수 처리 방식",
            ["svm_default", "normalized"],
            index=0 if st.session_state.score_method == 'svm_default' else 1,
            help="svm_default: 훈련과 동일한 방식 (권장)\nnormalized: 0-1 범위로 정규화"
        )
        
        if score_method != st.session_state.score_method:
            st.session_state.score_method = score_method
            st.rerun()
        
        # 점수 방식 설명
        if score_method == 'svm_default':
            st.info("🎯 **SVM 기본 방식**\n- 훈련과 동일한 방식\n- 임계값: 0\n- 양수: 이상, 음수: 정상")
        else:
            st.info("📊 **정규화 방식**\n- 0-1 범위 변환\n- 임계값: 0.5\n- 높을수록 이상")
        
        # 업데이트 간격 설정
        update_interval = st.slider(
            "업데이트 간격 (초)", 
            min_value=1, 
            max_value=5, 
            value=2,
            help="모니터링 데이터 업데이트 간격을 설정합니다"
        )
        
        # 현재 진행 상황
        if st.session_state.total_windows > 0:
            st.divider()
            st.subheader("📈 진행 상황")
            
            # 부품별 이상 비율
            for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
                if comp_name in st.session_state.anomaly_counts:
                    count = st.session_state.anomaly_counts[comp_name]
                    total = st.session_state.total_windows
                    rate = (count / total * 100) if total > 0 else 0
                    
                    comp_kr = comp_name.replace('cooler', '냉각기').replace('valve', '밸브').replace('pump', '펌프').replace('hydraulic', '유압시스템')
                    st.metric(comp_kr, f"{rate:.1f}%", f"{count}/{total}")
    
    # 메인 화면
    if not st.session_state.data_ready or not st.session_state.models_ready:
        # 상태 안내
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("💡 홈페이지에서 데이터를 로드하거나 models/ 폴더를 확인하세요.")
            
            # 체크리스트 표시
            st.markdown("### 📋 체크리스트")
            check1 = "✅" if st.session_state.data_ready else "❌"
            check2 = "✅" if st.session_state.models_ready else "❌"
            st.markdown(f"""
            - {check1} 센서 데이터 로드
            - {check2} 모델 로드
            """)
    else:
        # 모니터링 시작 안내 (모니터링이 시작되지 않았을 때만 표시)
        if not st.session_state.monitoring_active and st.session_state.total_windows == 0:
            st.markdown("### 🚀 모니터링 시작 안내")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info("""
                **🔍 실시간 모니터링을 시작하세요!**
                
                왼쪽 사이드바의 **▶️ 시작** 버튼을 클릭하면:
                - 📊 실시간 상태 모니터링
                - 📈 동적 그래프 업데이트  
                - 🔔 이상 탐지 알림
                - 📋 상세한 분석 결과
                
                가 표시됩니다.
                """)
                
        
        # 모니터링 화면
        if st.session_state.monitoring_active:
            try:
                window_data = get_window_data()
                if window_data:
                    anomaly_results = process_window(window_data)
                    
                    # 실시간 상태 표시
                    if anomaly_results:
                        st.markdown("### 🚨 실시간 상태")
                        cols = st.columns(4)
                        
                        comp_names = ['cooler', 'valve', 'pump', 'hydraulic']
                        comp_krs = ['냉각기', '밸브', '펌프', '유압시스템']
                        
                        for i, (comp_name, comp_kr) in enumerate(zip(comp_names, comp_krs)):
                            with cols[i]:
                                if comp_name in anomaly_results:
                                    result = anomaly_results[comp_name]
                                    score_label = "Raw" if st.session_state.score_method == 'svm_default' else "Norm"
                                    
                                    if result['is_anomaly']:
                                        st.error(f"🔴 {comp_kr}\n이상 탐지\n{score_label}: {result['display_score']:.3f}\n임계값: {result['threshold']:.1f}")
                                    else:
                                        st.success(f"🟢 {comp_kr}\n정상\n{score_label}: {result['display_score']:.3f}\n임계값: {result['threshold']:.1f}")
                
            except Exception as e:
                st.error(f"모니터링 중 오류 발생: {e}")
                st.session_state.monitoring_active = False
        
        # 상태 메트릭 및 그래프 (모니터링 시작 후에만 표시)
        if st.session_state.total_windows > 0:
            st.markdown("### 📊 모니터링 현황")
            create_status_metrics()
            
            # 그래프 표시
            st.markdown("### 📈 실시간 그래프")
            fig = create_monitoring_plots()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 그래프를 생성하기 위해 데이터를 수집하고 있습니다...")
            
            # 알림 히스토리
            if len(st.session_state.alert_history) > 0:
                st.markdown("### 🔔 최근 알림")
                
                # 최근 10개 알림 표시
                recent_alerts = list(st.session_state.alert_history)[:10]
                alert_data = []
                
                for alert in recent_alerts:
                    status = "🔴 이상" if alert['is_anomaly'] else "🟢 정상"
                    score_label = "Raw Score" if alert['score_method'] == 'svm_default' else "Norm Score"
                    
                    alert_data.append({
                        '시간': alert['time'],
                        '부품': alert['component'],
                        '상태': status,
                        score_label: f"{alert['display_score']:.3f}",
                        '임계값': f"{alert['threshold']:.3f}",
                        '방식': alert['score_method']
                    })
                
                if alert_data:
                    st.dataframe(pd.DataFrame(alert_data), use_container_width=True)
        
        # 자동 새로고침
        if st.session_state.monitoring_active:
            time.sleep(update_interval)
            st.rerun()

if __name__ == "__main__":
    main()