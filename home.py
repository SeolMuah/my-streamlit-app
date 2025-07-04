import streamlit as st
import time
from datetime import datetime
from util.data_loader import load_sensor_data, load_labels, load_models

# 페이지 설정
st.set_page_config(
    page_title="유압 시스템 모니터링",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .status-ready {
        border-left: 4px solid #10b981;
    }
    .status-error {
        border-left: 4px solid #ef4444;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 메인 페이지
st.title("유압 시스템 이상 탐지 모니터링")
st.markdown("---")

# 데이터 로드 상태 확인 및 로딩
def initialize_data_loading():
    """데이터 로딩 초기화 및 상태 확인"""
    
    # 세션 상태 확인
    if 'data_loading_completed' not in st.session_state:
        st.session_state.data_loading_completed = False
        st.session_state.data_loading_status = {}
    
    # 이미 로딩 완료된 경우 상태만 반환
    if st.session_state.data_loading_completed:
        return st.session_state.data_loading_status
    
    # 데이터 로딩 시작
    st.markdown("### 🔄 시스템 초기화 중...")
    st.markdown("---")
    
    # 로딩 상태 표시용 컨테이너
    loading_container = st.container()
    
    with loading_container:
        # 1. 센서 데이터 로딩
        st.markdown("#### 📊 센서 데이터 로딩")
        try:
            sensor_data = load_sensor_data(show_progress=True)
            sensor_count = len(sensor_data)
            sensor_list = list(sensor_data.keys())
            
            st.success(f"✅ 센서 데이터 로딩 완료! ({sensor_count}개 센서)")
            
        except Exception as e:
            st.error(f"❌ 센서 데이터 로딩 실패: {e}")
            sensor_count = 0
            sensor_list = []
        
        st.markdown("---")
        
        # 2. 라벨 데이터 로딩
        st.markdown("#### 📋 라벨 데이터 로딩")
        try:
            labels = load_labels(show_progress=True)
            cycle_count = len(labels) if labels is not None else 0
            
            st.success(f"✅ 라벨 데이터 로딩 완료! ({cycle_count}개 사이클)")
            
        except Exception as e:
            st.error(f"❌ 라벨 데이터 로딩 실패: {e}")
            cycle_count = 0
        
        st.markdown("---")
        
        # 3. 모델 파일 로딩
        st.markdown("#### 🤖 모델 파일 로딩")
        try:
            models, scalers, metadata = load_models(show_progress=True)
            
            if models is not None:
                model_count = len(models)
                st.success(f"✅ 모델 파일 로딩 완료! ({model_count}개 모델)")
                models_available = True
            else:
                st.warning("⚠️ 모델 파일을 찾을 수 없습니다. 모니터링 기능이 제한됩니다.")
                model_count = 0
                models_available = False
                
        except Exception as e:
            st.error(f"❌ 모델 파일 로딩 실패: {e}")
            model_count = 0
            models_available = False
        
        st.markdown("---")
        
        # 로딩 완료 상태 저장
        loading_status = {
            'status': 'completed',
            'sensor_count': sensor_count,
            'cycle_count': cycle_count,
            'model_count': model_count,
            'sensors': sensor_list,
            'models_available': models_available,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.data_loading_status = loading_status
        st.session_state.data_loading_completed = True
        
        # 최종 완료 메시지
        st.success("🎉 모든 데이터 로딩이 완료되었습니다!")
        
        # 자동으로 로딩 메시지 숨기기
        time.sleep(2)
        st.rerun()
    
    return loading_status

# 데이터 상태 확인 (간단한 버전)
@st.cache_data(ttl=60)  # 1분마다 갱신
def check_data_status_simple():
    """간단한 데이터 상태 확인 (로딩 표시 없음)"""
    try:
        sensor_data = load_sensor_data(show_progress=False)
        labels = load_labels(show_progress=False)
        models, scalers, metadata = load_models(show_progress=False)
        
        return {
            'status': 'ready',
            'sensor_count': len(sensor_data),
            'cycle_count': len(labels) if labels is not None else 0,
            'model_count': len(models) if models is not None else 0,
            'sensors': list(sensor_data.keys()),
            'models_available': models is not None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# 데이터 상태 표시
if not st.session_state.get('data_loading_completed', False):
    # 처음 로딩하는 경우
    data_status = initialize_data_loading()
else:
    # 이미 로딩 완료된 경우
    data_status = st.session_state.data_loading_status
    
    # 상태 정보 업데이트 (캐시된 버전)
    updated_status = check_data_status_simple()
    if updated_status['status'] == 'ready':
        data_status.update(updated_status)

# 데이터 상태 카드 표시
if data_status['status'] == 'completed' or data_status['status'] == 'ready':
    st.markdown(f"""
    <div class="status-card status-ready">
        <h3 style="color: #10b981; margin: 0 0 1rem 0;">✅ 시스템 준비 완료</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
            <div class="metric-card">
                <div class="metric-value">{data_status['sensor_count']}</div>
                <div class="metric-label">센서 수</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data_status['cycle_count']}</div>
                <div class="metric-label">사이클 수</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data_status.get('model_count', 0)}</div>
                <div class="metric-label">모델 수</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{'✓' if data_status.get('models_available', False) else '✗'}</div>
                <div class="metric-label">모델 상태</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="status-card status-error">
        <h3 style="color: #ef4444; margin: 0 0 1rem 0;">⚠️ 시스템 초기화 중</h3>
        <p style="margin: 0;">{data_status.get('error', '데이터를 로드하는 중입니다...')}</p>
    </div>
    """, unsafe_allow_html=True)

# 기능 소개
st.markdown("### 📊 주요 기능")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #3b82f6; margin: 0 0 0.5rem 0;">📈 데이터 분석 (EDA)</h4>
        <p style="margin: 0; font-size: 0.9rem;">• 센서 데이터 시각화</p>
        <p style="margin: 0; font-size: 0.9rem;">• 부품별 상태 분포</p>
        <p style="margin: 0; font-size: 0.9rem;">• 통계 분석 및 상관관계</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #059669; margin: 0 0 0.5rem 0;">🔍 실시간 모니터링</h4>
        <p style="margin: 0; font-size: 0.9rem;">• OneClass SVM 이상 탐지</p>
        <p style="margin: 0; font-size: 0.9rem;">• 실시간 센서 데이터 분석</p>
        <p style="margin: 0; font-size: 0.9rem;">• 부품별 상태 모니터링</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #dc2626; margin: 0 0 0.5rem 0;">🚨 이상 탐지</h4>
        <p style="margin: 0; font-size: 0.9rem;">• 머신러닝 기반 분석</p>
        <p style="margin: 0; font-size: 0.9rem;">• 부품별 이상 상태 감지</p>
        <p style="margin: 0; font-size: 0.9rem;">• 실시간 알림 시스템</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 시스템 정보
st.markdown("### 🔧 시스템 정보")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **모니터링 대상 부품:**
    - 냉각기 (Cooler)
    - 밸브 (Valve)  
    - 펌프 (Pump)
    - 유압 시스템 (Hydraulic)
    """)

with col2:
    st.markdown("""
    **센서 유형:**
    - 압력 센서 (PS1-PS6): 100Hz
    - 온도 센서 (TS1-TS4): 1Hz
    - 유량 센서 (FS1-FS2): 10Hz
    - 전력 센서 (EPS1): 100Hz
    - 진동 센서 (VS1): 1Hz
    """)

# 사용 방법
st.markdown("### 🚀 사용 방법")
st.markdown("""
1. **사이드바**에서 원하는 분석 페이지를 선택하세요
2. **데이터 분석**: 센서 데이터 탐색 및 시각화
3. **실시간 모니터링**: 이상 탐지 모니터링 시작
4. **시스템 상태**: 우측 사이드바에서 확인 가능
""")

# 푸터
st.markdown("---")
st.caption("유압 시스템 이상 탐지 모니터링 시스템 | 머신러닝 기반 실시간 분석")

# 사이드바에 데이터 정보 표시
with st.sidebar:
    st.header("📊 시스템 정보")
    
    if data_status['status'] in ['completed', 'ready']:
        st.success("✅ 시스템 준비 완료")
        
        # 메트릭 표시
        col1, col2 = st.columns(2)
        with col1:
            st.metric("센서 수", data_status['sensor_count'])
            st.metric("모델 수", data_status.get('model_count', 0))
        with col2:
            st.metric("사이클 수", data_status['cycle_count'])
            st.metric("로딩 시간", data_status.get('timestamp', 'N/A').split(' ')[1] if data_status.get('timestamp') else 'N/A')
        
        # 상태 표시
        st.markdown("**🔧 기능 상태**")
        st.markdown(f"• 📊 EDA: ✅ 사용 가능")
        st.markdown(f"• 🔍 모니터링: {'✅ 사용 가능' if data_status.get('models_available', False) else '❌ 모델 필요'}")
        st.markdown(f"• 💬 AI 채팅: ✅ 사용 가능")
        
        # 캐시 제어
        st.markdown("---")
        st.markdown("**🔄 데이터 관리**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 새로고침"):
                st.cache_data.clear()
                st.session_state.data_loading_completed = False
                st.rerun()
        
        with col2:
            if st.button("📊 상태 확인"):
                updated_status = check_data_status_simple()
                if updated_status['status'] == 'ready':
                    st.success("✅ 상태 양호")
                else:
                    st.error("❌ 상태 불량")
        
        # 세부 정보
        with st.expander("🔍 상세 정보"):
            st.markdown("**센서 목록:**")
            for i, sensor in enumerate(data_status['sensors']):
                st.text(f"{i+1:2d}. {sensor}")
            
            st.markdown("**시스템 상태:**")
            st.text(f"마지막 업데이트: {data_status.get('timestamp', 'N/A')}")
            st.text(f"캐시 상태: 활성")
    else:
        st.error("❌ 시스템 오류")
        st.markdown(f"**오류 내용:**\n{data_status.get('error', '알 수 없는 오류')}")
        
        if st.button("🔄 재시도"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.data_loading_completed = False
            st.rerun()
    
    # 시스템 정보
    st.markdown("---")
    st.markdown("**💡 도움말**")
    st.markdown("""
    - 🔄 새로고침: 모든 데이터 다시 로드
    - 📊 상태 확인: 현재 상태만 확인
    - 🔍 상세 정보: 센서 목록 및 상태
    """)
    
    # 성능 정보
    with st.expander("⚡ 성능 정보"):
        st.markdown("**캐시 상태:**")
        st.text("• 센서 데이터: 1시간 캐시")
        st.text("• 라벨 데이터: 1시간 캐시")
        st.text("• 모델 파일: 1시간 캐시")
        st.text("• 상태 체크: 1분 캐시")