import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from util.data_loader import load_sensor_data, load_labels, get_sensor_time_axis, SENSOR_FREQUENCIES

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 탐색적 데이터 분석 (EDA)")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("🎛️ 분석 설정")
    analysis_type = st.selectbox(
        "분석 유형",
        ["센서 데이터 시각화", "부품 상태 분포", "통계 요약", "센서 상관관계", "이상 데이터 탐지"]
    )
    
    # 데이터 새로고침 버튼
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

# 데이터 로드 (캐시된 함수 사용)
try:
    # 페이지에서는 진행 상황 표시 안 함 (이미 홈에서 로딩 완료)
    sensor_data = load_sensor_data(show_progress=False)
    labels = load_labels(show_progress=False)
    
    # 데이터 로드 성공 메시지
    st.success(f"✅ 데이터 로드 완료! 센서 {len(sensor_data)}개, 사이클 {len(labels)}개")
    
    if analysis_type == "센서 데이터 시각화":
        st.header("📈 센서 데이터 시각화")
        
        # 센서 선택
        available_sensors = list(sensor_data.keys())
        selected_sensors = st.multiselect(
            "시각화할 센서 선택",
            available_sensors,
            default=available_sensors[:3] if len(available_sensors) >= 3 else available_sensors
        )
        
        if selected_sensors:
            # 사이클 선택
            max_cycles = len(labels) - 1
            cycle_idx = st.slider("사이클 선택", 0, max_cycles, 0)
            
            # 그래프 생성
            fig = make_subplots(
                rows=len(selected_sensors), 
                cols=1,
                subplot_titles=[f"{sensor} - 주파수: {SENSOR_FREQUENCIES.get(sensor, 1)}Hz" for sensor in selected_sensors],
                shared_xaxes=True,
                vertical_spacing=0.05
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            for idx, sensor in enumerate(selected_sensors):
                if sensor in sensor_data:
                    # 센서별 시간축 생성
                    time_axis = get_sensor_time_axis(sensor)
                    
                    # 해당 사이클의 데이터
                    data = sensor_data[sensor].iloc[cycle_idx].values
                    
                    # 시간축 길이에 맞게 데이터 조정
                    if len(data) > len(time_axis):
                        data = data[:len(time_axis)]
                    elif len(data) < len(time_axis):
                        time_axis = time_axis[:len(data)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=data, 
                            mode='lines', 
                            name=sensor,
                            line=dict(color=colors[idx % len(colors)], width=2),
                            hovertemplate=f'<b>{sensor}</b><br>' +
                                        'Time: %{x:.2f}s<br>' +
                                        'Value: %{y:.2f}<br>' +
                                        '<extra></extra>'
                        ),
                        row=idx+1, col=1
                    )
                    
                    # Y축 제목 추가
                    fig.update_yaxes(title_text=f"{sensor} 값", row=idx+1, col=1)
            
            fig.update_layout(
                height=300*len(selected_sensors), 
                showlegend=False,
                title_text=f"센서 데이터 시각화 - 사이클 {cycle_idx} (60초 구간)"
            )
            fig.update_xaxes(title_text="시간 (초)", row=len(selected_sensors), col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 선택된 센서들의 통계 정보
            st.subheader("📊 선택된 센서 통계")
            
            col1, col2 = st.columns(2)
            
            with col1:
                stats_data = []
                for sensor in selected_sensors:
                    if sensor in sensor_data:
                        data = sensor_data[sensor].iloc[cycle_idx].values
                        freq = SENSOR_FREQUENCIES.get(sensor, 1)
                        stats_data.append({
                            '센서': sensor,
                            '주파수': f"{freq}Hz",
                            '샘플수': len(data),
                            '평균': f"{np.mean(data):.2f}",
                            '표준편차': f"{np.std(data):.2f}",
                            '최대값': f"{np.max(data):.2f}",
                            '최소값': f"{np.min(data):.2f}"
                        })
                
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            with col2:
                # 센서별 주파수 정보 표시
                st.markdown("**센서별 주파수 정보**")
                st.markdown("""
                - **압력 센서 (PS1-PS6)**: 100Hz
                - **전력 센서 (EPS1)**: 100Hz  
                - **유량 센서 (FS1-FS2)**: 10Hz
                - **온도 센서 (TS1-TS4)**: 1Hz
                - **진동 센서 (VS1)**: 1Hz
                - **효율 센서 (CE, CP, SE)**: 1Hz
                """)
                
                st.info("모든 센서는 60초 사이클로 동기화되어 표시됩니다.")
        
        else:
            st.warning("분석할 센서를 선택해주세요.")
    
    elif analysis_type == "부품 상태 분포":
        st.header("📊 부품 상태 분포")
        
        # 부품별 상태 분포 계산
        components = ['cooler', 'valve', 'pump', 'hydraulic']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[comp.upper() for comp in components],
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        colors_list = [
            ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
            ['#ff6666', '#6699ff', '#66ff66', '#ffaa66'],
            ['#ff3333', '#3366ff', '#33ff33', '#ff8833'],
            ['#cc0000', '#0033cc', '#00cc00', '#cc5500']
        ]
        
        for idx, comp in enumerate(components):
            values = labels[comp].value_counts().sort_index()
            row, col = positions[idx]
            
            fig.add_trace(
                go.Pie(
                    labels=[f"상태 {label}" for label in values.index], 
                    values=values.values, 
                    name=comp,
                    marker_colors=colors_list[idx][:len(values)]
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="부품별 상태 분포")
        st.plotly_chart(fig, use_container_width=True)
        
        # 상태별 설명
        st.subheader("📋 부품 상태 설명")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Cooler (냉각기):**
            - 3: 고장
            - 20: 효율 감소
            - 100: 정상
            
            **Valve (밸브):**
            - 73: 고장
            - 80: 심각
            - 90: 약간 지연
            - 100: 정상
            """)
        
        with col2:
            st.markdown("""
            **Pump (펌프):**
            - 0: 정상
            - 1: 약한 누출
            - 2: 심각한 누출
            
            **Hydraulic (유압):**
            - 90: 고장
            - 100: 낮음
            - 115: 약간 낮음
            - 130: 정상
            """)
        
        # 전체 상태 요약
        st.subheader("📈 전체 상태 요약")
        
        # 정상 상태 비율 계산
        normal_states = {
            'cooler': 100,
            'valve': 100,
            'pump': 0,
            'hydraulic': 130
        }
        
        summary_data = []
        for comp in components:
            total = len(labels)
            normal_count = sum(labels[comp] == normal_states[comp])
            normal_ratio = (normal_count / total) * 100
            
            summary_data.append({
                '부품': comp.upper(),
                '전체 사이클': total,
                '정상 사이클': normal_count,
                '정상 비율': f"{normal_ratio:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    elif analysis_type == "통계 요약":
        st.header("📊 센서별 통계 요약")
        
        # 사이클 선택
        cycle_options = ['전체 평균', '특정 사이클']
        cycle_choice = st.radio("분석 대상", cycle_options)
        
        if cycle_choice == '특정 사이클':
            cycle_idx = st.slider("사이클 선택", 0, len(labels)-1, 0)
            st.info(f"사이클 {cycle_idx} 분석 중...")
        
        # 통계 계산
        stats_data = []
        
        for sensor_name, data in sensor_data.items():
            if cycle_choice == '전체 평균':
                # 전체 사이클의 평균 통계
                all_means = []
                all_stds = []
                all_maxs = []
                all_mins = []
                
                for i in range(len(data)):
                    values = data.iloc[i].values[:1000]  # 첫 1000개 포인트만 사용
                    all_means.append(np.mean(values))
                    all_stds.append(np.std(values))
                    all_maxs.append(np.max(values))
                    all_mins.append(np.min(values))
                
                stats_data.append({
                    '센서': sensor_name,
                    '평균': f"{np.mean(all_means):.2f}",
                    '표준편차': f"{np.mean(all_stds):.2f}",
                    '최대값': f"{np.max(all_maxs):.2f}",
                    '최소값': f"{np.min(all_mins):.2f}",
                    '전체 범위': f"{np.max(all_maxs) - np.min(all_mins):.2f}"
                })
            else:
                # 특정 사이클 통계
                values = data.iloc[cycle_idx].values[:1000]
                stats_data.append({
                    '센서': sensor_name,
                    '평균': f"{np.mean(values):.2f}",
                    '표준편차': f"{np.std(values):.2f}",
                    '최대값': f"{np.max(values):.2f}",
                    '최소값': f"{np.min(values):.2f}",
                    '범위': f"{np.max(values) - np.min(values):.2f}"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # 센서 타입별 비교
        st.subheader("🔍 센서 타입별 비교")
        
        sensor_types = {
            'PS': [s for s in sensor_data.keys() if s.startswith('PS')],
            'TS': [s for s in sensor_data.keys() if s.startswith('TS')],
            'FS': [s for s in sensor_data.keys() if s.startswith('FS')],
            'EPS': [s for s in sensor_data.keys() if s.startswith('EPS')],
            'VS': [s for s in sensor_data.keys() if s.startswith('VS')]
        }
        
        # 존재하는 센서 타입만 필터링
        sensor_types = {k: v for k, v in sensor_types.items() if v}
        
        if sensor_types:
            cols = st.columns(len(sensor_types))
            
            for idx, (sensor_type, sensors) in enumerate(sensor_types.items()):
                with cols[idx]:
                    type_means = []
                    for sensor in sensors:
                        sensor_stats = stats_df[stats_df['센서'] == sensor]
                        if not sensor_stats.empty:
                            mean_val = float(sensor_stats['평균'].iloc[0])
                            type_means.append(mean_val)
                    
                    if type_means:
                        fig = go.Figure(data=[
                            go.Bar(x=sensors, y=type_means, 
                                  marker_color=f'rgba({50 + idx*50}, {100 + idx*30}, {200 - idx*20}, 0.8)')
                        ])
                        fig.update_layout(
                            title=f"{sensor_type} 센서 평균값",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "센서 상관관계":
        st.header("🔗 센서 간 상관관계 분석")
        
        # 상관관계 계산을 위한 데이터 준비
        cycle_idx = st.slider("분석할 사이클 선택", 0, len(labels)-1, 0)
        
        # 각 센서의 대표값 계산 (평균값 사용)
        sensor_values = {}
        for sensor_name, data in sensor_data.items():
            values = data.iloc[cycle_idx].values[:1000]
            sensor_values[sensor_name] = np.mean(values)
        
        # 여러 사이클의 상관관계 분석
        correlation_data = []
        sample_size = min(50, len(labels))  # 최대 50개 사이클 사용
        
        for i in range(sample_size):
            cycle_data = {}
            for sensor_name, data in sensor_data.items():
                values = data.iloc[i].values[:1000]
                cycle_data[sensor_name] = np.mean(values)
            correlation_data.append(cycle_data)
        
        # 상관관계 DataFrame 생성
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=f"센서 간 상관관계 매트릭스 (사이클 {sample_size}개 기준)",
            height=600,
            width=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 높은 상관관계 센서 쌍 찾기
        st.subheader("🔍 높은 상관관계 센서 쌍")
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 상관계수 0.7 이상
                    high_corr_pairs.append({
                        '센서1': correlation_matrix.columns[i],
                        '센서2': correlation_matrix.columns[j],
                        '상관계수': f"{corr_val:.3f}",
                        '관계': '강한 양의 상관관계' if corr_val > 0 else '강한 음의 상관관계'
                    })
        
        if high_corr_pairs:
            st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
        else:
            st.info("상관계수 0.7 이상인 센서 쌍이 없습니다.")
    
    elif analysis_type == "이상 데이터 탐지":
        st.header("🚨 이상 데이터 탐지")
        
        # 이상 상태인 사이클 찾기
        normal_states = {
            'cooler': 100,
            'valve': 100,
            'pump': 0,
            'hydraulic': 130
        }
        
        # 이상 사이클 식별
        anomaly_cycles = []
        for i in range(len(labels)):
            anomaly_components = []
            for comp in ['cooler', 'valve', 'pump', 'hydraulic']:
                if labels[comp].iloc[i] != normal_states[comp]:
                    anomaly_components.append(comp)
            
            if anomaly_components:
                anomaly_cycles.append({
                    '사이클': i,
                    '이상 부품': ', '.join(anomaly_components),
                    '이상 개수': len(anomaly_components)
                })
        
        st.subheader("📋 이상 사이클 목록")
        
        if anomaly_cycles:
            anomaly_df = pd.DataFrame(anomaly_cycles)
            st.dataframe(anomaly_df, use_container_width=True)
            
            # 이상 사이클 선택해서 상세 분석
            selected_anomaly = st.selectbox(
                "상세 분석할 이상 사이클 선택",
                anomaly_df['사이클'].tolist(),
                format_func=lambda x: f"사이클 {x} ({anomaly_df[anomaly_df['사이클']==x]['이상 부품'].iloc[0]})"
            )
            
            if selected_anomaly is not None:
                st.subheader(f"🔍 사이클 {selected_anomaly} 상세 분석")
                
                # 해당 사이클의 센서 데이터 시각화
                selected_sensors = st.multiselect(
                    "분석할 센서 선택",
                    list(sensor_data.keys()),
                    default=list(sensor_data.keys())[:4]
                )
                
                if selected_sensors:
                    fig = make_subplots(
                        rows=len(selected_sensors), 
                        cols=1,
                        subplot_titles=[f"{sensor} - 사이클 {selected_anomaly}" for sensor in selected_sensors],
                        shared_xaxes=True
                    )
                    
                    for idx, sensor in enumerate(selected_sensors):
                        if sensor in sensor_data:
                            data = sensor_data[sensor].iloc[selected_anomaly].values[:1000]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(data))),
                                    y=data,
                                    mode='lines',
                                    name=sensor,
                                    line=dict(color='red', width=2)
                                ),
                                row=idx+1, col=1
                            )
                    
                    fig.update_layout(
                        height=300*len(selected_sensors),
                        showlegend=False,
                        title_text=f"이상 사이클 {selected_anomaly} 센서 데이터"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 해당 사이클의 부품 상태 정보
                st.subheader("🔧 부품 상태 정보")
                
                cycle_info = []
                for comp in ['cooler', 'valve', 'pump', 'hydraulic']:
                    current_state = labels[comp].iloc[selected_anomaly]
                    normal_state = normal_states[comp]
                    is_normal = current_state == normal_state
                    
                    cycle_info.append({
                        '부품': comp.upper(),
                        '현재 상태': current_state,
                        '정상 상태': normal_state,
                        '상태': '정상' if is_normal else '이상',
                        '상태 표시': '✅' if is_normal else '❌'
                    })
                
                st.dataframe(pd.DataFrame(cycle_info), use_container_width=True)
        
        else:
            st.success("🎉 모든 사이클이 정상 상태입니다!")
        
        # 전체 이상 통계
        st.subheader("📊 전체 이상 통계")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("전체 사이클 수", len(labels))
            st.metric("이상 사이클 수", len(anomaly_cycles))
        
        with col2:
            anomaly_rate = (len(anomaly_cycles) / len(labels)) * 100 if len(labels) > 0 else 0
            st.metric("이상 비율", f"{anomaly_rate:.1f}%")
            st.metric("정상 사이클 수", len(labels) - len(anomaly_cycles))

except Exception as e:
    st.error(f"❌ 데이터 로드 중 오류 발생: {str(e)}")
    st.markdown("""
    **해결 방법:**
    1. `data` 폴더에 센서 데이터 파일이 있는지 확인하세요
    2. 사이드바에서 '🔄 데이터 새로고침' 버튼을 클릭하세요
    3. 문제가 지속되면 시뮬레이션 데이터가 자동으로 생성됩니다
    """)
    
    # 오류 발생 시 캐시 초기화 옵션 제공
    if st.button("🔄 캐시 초기화 및 재시도"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()