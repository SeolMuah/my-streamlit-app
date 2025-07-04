import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import time
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 상수 정의
# =============================================================================
MAX_ROWS = 100 #데이터 100개 행만 읽도록 함 (센서 및 라벨)

# 센서별 주파수 정보 (Hz)
SENSOR_FREQUENCIES = {
    'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
    'EPS1': 100,
    'FS1': 10, 'FS2': 10,
    'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
    'VS1': 1,
    'CE': 1, 'CP': 1, 'SE': 1
}

# 센서별 상세 정보
SENSOR_INFO = {
    'PS1': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'PS2': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'PS3': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'PS4': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'PS5': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'PS6': {'freq': 100, 'type': 'pressure', 'unit': 'bar', 'range': (140, 160)},
    'EPS1': {'freq': 100, 'type': 'motor_power', 'unit': 'W', 'range': (2000, 2500)},
    'FS1': {'freq': 10, 'type': 'flow', 'unit': 'L/min', 'range': (8, 12)},
    'FS2': {'freq': 10, 'type': 'flow', 'unit': 'L/min', 'range': (8, 12)},
    'TS1': {'freq': 1, 'type': 'temperature', 'unit': '°C', 'range': (35, 40)},
    'TS2': {'freq': 1, 'type': 'temperature', 'unit': '°C', 'range': (35, 40)},
    'TS3': {'freq': 1, 'type': 'temperature', 'unit': '°C', 'range': (35, 40)},
    'TS4': {'freq': 1, 'type': 'temperature', 'unit': '°C', 'range': (35, 40)},
    'VS1': {'freq': 1, 'type': 'vibration', 'unit': 'mm/s', 'range': (0.5, 1.0)},
    'CE': {'freq': 1, 'type': 'cooling_efficiency', 'unit': '%', 'range': (0.8, 1.2)},
    'CP': {'freq': 1, 'type': 'cooling_power', 'unit': 'kW', 'range': (0.8, 1.2)},
    'SE': {'freq': 1, 'type': 'efficiency_factor', 'unit': '%', 'range': (0.8, 1.2)}
}

# 부품별 상태 정의
COMPONENT_STATES = {
    'cooler': {3: 'close_to_failure', 20: 'reduced_efficiency', 100: 'full_efficiency'},
    'valve': {73: 'optimal_switching', 80: 'small_lag', 90: 'severe_lag', 100: 'close_to_failure'},
    'pump': {0: 'no_leakage', 1: 'weak_leakage', 2: 'severe_leakage'},
    'hydraulic': {90: 'close_to_failure', 100: 'optimal_pressure', 115: 'slightly_reduced', 130: 'severely_reduced'}
}

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_sensor_time_axis(sensor_name, cycle_duration=60):
    """센서별 시간축 생성"""
    freq = SENSOR_FREQUENCIES.get(sensor_name, 1)
    num_samples = freq * cycle_duration
    return np.linspace(0, cycle_duration, num_samples)

def safe_calculate_feature(values, feature_func, default_value=0.0):
    """안전한 특징 계산 (NaN 처리)"""
    try:
        if len(values) < 2:
            return default_value
        result = feature_func(values)
        return result if np.isfinite(result) else default_value
    except Exception:
        return default_value

def generate_dummy_cycle_data(sensor_name, cycle_length):
    """센서별 더미 사이클 데이터 생성"""
    sensor_info = SENSOR_INFO.get(sensor_name, {'range': (0, 1)})
    base_value = np.random.uniform(*sensor_info['range'])
    noise_level = base_value * 0.03
    return base_value + np.random.normal(0, noise_level, cycle_length)

# =============================================================================
# 데이터 로더 클래스
# =============================================================================

class HydraulicDataLoader:
    """유압 시스템 데이터 로더"""
    
    def __init__(self, data_folder_path='data', window_seconds=10, stride_seconds=10):
        self.data_folder_path = data_folder_path
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.sensor_data = {}
        self.profile_data = None
        self.current_cycle_index = 0
        self.num_cycles = 0
        self.is_dummy_data = False

    def _load_all_data(self):
        """모든 센서 데이터와 프로파일 로드"""
        if not os.path.exists(self.data_folder_path):
            st.warning(f"데이터 폴더 '{self.data_folder_path}'를 찾을 수 없습니다. 시뮬레이션 데이터를 사용합니다.")
            self._generate_dummy_data()
            return

        try:
            with st.spinner("데이터 로드 중..."):
                self._load_sensor_files()
                self._load_profile_file()
                
                if not self.sensor_data:
                    self._generate_dummy_data()
                    st.warning("실제 데이터를 로드할 수 없어 시뮬레이션 데이터를 사용합니다.")
                else:
                    st.success(f"데이터 로드 완료! 총 {self.num_cycles}개 사이클")
                    
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            self._generate_dummy_data()

    def _load_sensor_files(self):
        """센서 파일들 로드"""
        sensor_names = list(SENSOR_INFO.keys())
        progress_bar = st.progress(0, text="센서 데이터 파일 읽는 중...")
        
        for i, sensor_name in enumerate(sensor_names):
            file_path = os.path.join(self.data_folder_path, f"{sensor_name}.txt")
            progress_bar.progress(
                int(((i + 1) / len(sensor_names)) * 80), 
                text=f"{sensor_name}.txt 로드 중..."
            )
            
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path, sep='\t', header=None)
                    self.sensor_data[sensor_name] = data
                except Exception as e:
                    st.warning(f"센서 '{sensor_name}' 파일 로드 실패: {e}")
            else:
                st.warning(f"센서 '{sensor_name}' 파일을 찾을 수 없습니다.")
        
        progress_bar.empty()

    def _load_profile_file(self):
        """프로파일 파일 로드"""
        profile_path = os.path.join(self.data_folder_path, 'profile.txt')
        
        if os.path.exists(profile_path):
            try:
                self.profile_data = pd.read_csv(
                    profile_path, 
                    sep='\t', 
                    header=None,
                    names=['cooler', 'valve', 'pump', 'hydraulic', 'stable']
                )
                self.num_cycles = len(self.profile_data)
            except Exception as e:
                st.warning(f"프로파일 데이터 로드 실패: {e}")
        else:
            st.warning("프로파일 데이터를 찾을 수 없습니다.")

    def _generate_dummy_data(self):
        """더미 데이터 생성"""
        self.num_cycles = 100
        self.is_dummy_data = True
        
        # 센서 데이터 생성
        for sensor_name, info in SENSOR_INFO.items():
            freq = info['freq']
            cycle_length = freq * 60  # 60초 사이클
            
            dummy_data = []
            for cycle in range(self.num_cycles):
                cycle_data = generate_dummy_cycle_data(sensor_name, cycle_length)
                dummy_data.append(cycle_data)
            
            self.sensor_data[sensor_name] = pd.DataFrame(dummy_data)
        
        # 프로파일 데이터 생성
        self.profile_data = pd.DataFrame({
            'cooler': np.random.choice([3, 20, 100], self.num_cycles, p=[0.1, 0.2, 0.7]),
            'valve': np.random.choice([73, 80, 90, 100], self.num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
            'pump': np.random.choice([0, 1, 2], self.num_cycles, p=[0.7, 0.2, 0.1]),
            'hydraulic': np.random.choice([90, 100, 115, 130], self.num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
            'stable': np.ones(self.num_cycles, dtype=int)
        })

    def get_next_window(self):
        """다음 10초 윈도우 데이터 반환"""
        if self.num_cycles == 0:
            return None, None
        
        # 사이클 인덱스 순환
        if self.current_cycle_index >= self.num_cycles:
            self.current_cycle_index = 0
        
        cycle_idx = self.current_cycle_index
        window_start = np.random.choice([0, 10, 20, 30, 40, 50])  # 랜덤 시작 위치
        window_end = window_start + self.window_seconds
        
        window_data = {}
        
        # 각 센서의 윈도우 데이터 추출
        for sensor_name, sensor_df in self.sensor_data.items():
            freq = SENSOR_INFO[sensor_name]['freq']
            start_idx = int(window_start * freq)
            end_idx = int(window_end * freq)
            
            cycle_data = sensor_df.iloc[cycle_idx].values
            window_data[sensor_name] = cycle_data[start_idx:end_idx]
        
        # 라벨 정보
        cycle_labels = self.profile_data.iloc[cycle_idx] if self.profile_data is not None else None
        
        self.current_cycle_index += 1
        return window_data, cycle_labels

# =============================================================================
# 특징 추출 함수
# =============================================================================

def extract_features_from_window(window_data):
    """윈도우 데이터에서 특징 추출 (개선된 버전)"""
    features = {}
    
    for sensor_name, sensor_values in window_data.items():
        values = np.array(sensor_values)
        
        # 유효한 값만 선택
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) < 2:
            # 기본값으로 채우기
            feature_defaults = {
                'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0,
                'rms': 0.0, 'peak2peak': 0.0, 'skew': 0.0, 'kurtosis': 0.0
            }
            for feature_name, default_val in feature_defaults.items():
                features[f'{sensor_name}_{feature_name}'] = default_val
            continue
        
        # 기본 통계 특징
        features[f'{sensor_name}_mean'] = safe_calculate_feature(valid_values, np.mean)
        features[f'{sensor_name}_std'] = safe_calculate_feature(valid_values, np.std)
        features[f'{sensor_name}_max'] = safe_calculate_feature(valid_values, np.max)
        features[f'{sensor_name}_min'] = safe_calculate_feature(valid_values, np.min)
        features[f'{sensor_name}_rms'] = safe_calculate_feature(
            valid_values, lambda x: np.sqrt(np.mean(x**2))
        )
        features[f'{sensor_name}_peak2peak'] = safe_calculate_feature(
            valid_values, lambda x: np.max(x) - np.min(x)
        )
        
        # 고차 통계 특징
        if len(valid_values) > 2:
            features[f'{sensor_name}_skew'] = safe_calculate_feature(valid_values, skew)
            features[f'{sensor_name}_kurtosis'] = safe_calculate_feature(valid_values, kurtosis)
        else:
            features[f'{sensor_name}_skew'] = 0.0
            features[f'{sensor_name}_kurtosis'] = 0.0
    
    return features

# =============================================================================
# 이상 탐지 함수
# =============================================================================

def detect_anomaly(features, models, scalers, metadata):
    """각 부품별 이상 탐지 (개선된 버전)"""
    # 특징 DataFrame 생성
    features_df = pd.DataFrame([features])
    
    # 메타데이터에서 특징 컬럼 정보 가져오기
    feature_cols = metadata.get('feature_columns', [])
    
    # 필요한 특징만 선택하고 누락된 특징은 0으로 채움
    features_selected = features_df.reindex(columns=feature_cols, fill_value=0.0)
    
    # NaN 및 inf 값 처리
    features_selected = features_selected.fillna(0.0)
    features_selected = features_selected.replace([np.inf, -np.inf], 0.0)
    
    results = {}
    
    for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
        try:
            # 스케일링
            X_scaled = scalers[comp_name].transform(features_selected)
            
            # 스케일링 후 NaN/inf 처리
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 예측
            prediction = models[comp_name].predict(X_scaled)[0]
            score = models[comp_name].decision_function(X_scaled)[0]
            
            # 점수 검증
            if not np.isfinite(score):
                score = 0.0
            
            results[comp_name] = {
                'is_anomaly': prediction == -1,
                'score': float(score),
                'confidence': abs(float(score))
            }
            
        except Exception as e:
            st.error(f"{comp_name} 예측 중 오류 발생: {e}")
            results[comp_name] = {
                'is_anomaly': False,
                'score': 0.0,
                'confidence': 0.0
            }
    
    return results

# =============================================================================
# 더미 데이터 생성 함수
# =============================================================================

def generate_dummy_sensor_data():
    """더미 센서 데이터 생성 (캐시용)"""
    sensor_data = {}
    num_cycles = 100
    
    for sensor_name, info in SENSOR_INFO.items():
        freq = info['freq']
        cycle_length = freq * 60  # 60초 사이클
        
        dummy_data = []
        for cycle in range(num_cycles):
            cycle_data = generate_dummy_cycle_data(sensor_name, cycle_length)
            dummy_data.append(cycle_data)
        
        sensor_data[sensor_name] = pd.DataFrame(dummy_data)
    
    return sensor_data

def generate_dummy_labels():
    """더미 라벨 데이터 생성 (캐시용)"""
    num_cycles = 100
    
    return pd.DataFrame({
        'cooler': np.random.choice([3, 20, 100], num_cycles, p=[0.1, 0.2, 0.7]),
        'valve': np.random.choice([73, 80, 90, 100], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'pump': np.random.choice([0, 1, 2], num_cycles, p=[0.7, 0.2, 0.1]),
        'hydraulic': np.random.choice([90, 100, 115, 130], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'stable': np.ones(num_cycles, dtype=int)
    })

# =============================================================================
# 캐시된 데이터 로더 함수들
# =============================================================================

@st.cache_resource(ttl=3600)
def get_data_loader(data_folder_path='data'):
    """데이터 로더 인스턴스 생성 (캐시됨)"""
    loader = HydraulicDataLoader(data_folder_path)
    loader._load_all_data()
    return loader

@st.cache_data(ttl=3600)
def load_sensor_data(data_folder_path='data', show_progress=True):
    """센서 데이터 로드 (캐시됨)"""
    sensor_list = list(SENSOR_INFO.keys())
    sensor_data = {}
    
    if show_progress:
        progress_bar = st.progress(0, text="센서 데이터 파일을 확인하는 중입니다...")
        status_placeholder = st.empty()
        
        # 파일 존재 확인
        existing_files = [
            sensor for sensor in sensor_list 
            if os.path.exists(os.path.join(data_folder_path, f'{sensor}.txt'))
        ]
        
        if not existing_files:
            progress_bar.progress(100, text="시뮬레이션 데이터를 생성합니다...")
            status_placeholder.warning("📁 실제 센서 파일을 찾을 수 없습니다. 시뮬레이션 데이터를 생성합니다.")
            time.sleep(1)
            sensor_data = generate_dummy_sensor_data()
            progress_bar.empty()
            status_placeholder.empty()
            return sensor_data
        
        # 파일 로딩
        for i, sensor in enumerate(existing_files):
            file_path = os.path.join(data_folder_path, f'{sensor}.txt')
            progress_percent = int((i / len(existing_files)) * 100)
            progress_bar.progress(progress_percent, text=f"📊 {sensor}.txt 파일 로딩 중... ({i+1}/{len(existing_files)})")
            
            try:
                status_placeholder.info(f"🔄 {sensor}.txt 파일을 읽고 있습니다...")
                data = pd.read_csv(file_path, sep='\t', header=None, nrows=MAX_ROWS)
                sensor_data[sensor] = data
                status_placeholder.success(f"✅ {sensor}.txt 파일 로드 완료! ({len(data)} 행)")
                time.sleep(0.1)
            except Exception as e:
                status_placeholder.error(f"❌ {sensor}.txt 파일 로드 실패: {e}")
                time.sleep(0.5)
        
        progress_bar.progress(100, text=f"✅ 센서 데이터 로딩 완료!")
        
        if not sensor_data:
            status_placeholder.warning("시뮬레이션 데이터를 생성합니다.")
            sensor_data = generate_dummy_sensor_data()
        else:
            status_placeholder.success(f"🎉 총 {len(sensor_data)}개의 센서 데이터 파일이 성공적으로 로드되었습니다!")
        
        time.sleep(1)
        progress_bar.empty()
        status_placeholder.empty()
    else:
        # 진행 상황 표시 없이 로드
        for sensor in sensor_list:
            file_path = os.path.join(data_folder_path, f'{sensor}.txt')
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path, sep='\t', header=None, nrows=MAX_ROWS)
                    sensor_data[sensor] = data
                except Exception as e:
                    st.warning(f"센서 {sensor} 데이터 로드 실패: {e}")
        
        if not sensor_data:
            sensor_data = generate_dummy_sensor_data()
    
    return sensor_data

@st.cache_data(ttl=3600)
def load_labels(data_folder_path='data', show_progress=True):
    """라벨 데이터 로드 (캐시됨)"""
    profile_path = os.path.join(data_folder_path, 'profile.txt')
    
    if show_progress:
        progress_bar = st.progress(0, text="라벨 데이터 파일을 확인하는 중입니다...")
        status_placeholder = st.empty()
        
        if os.path.exists(profile_path):
            try:
                progress_bar.progress(50, text="📋 profile.txt 파일 읽는 중...")
                status_placeholder.info("🔄 profile.txt 파일을 읽고 있습니다...")
                
                labels = pd.read_csv(profile_path, sep='\t', header=None,
                                   names=['cooler', 'valve', 'pump', 'hydraulic', 'stable'])
                
                progress_bar.progress(100, text="✅ 라벨 데이터 로딩 완료!")
                status_placeholder.success(f"✅ profile.txt 파일 로드 완료! ({len(labels)} 라벨)")
                
                time.sleep(0.5)
                progress_bar.empty()
                status_placeholder.empty()
                return labels
                
            except Exception as e:
                progress_bar.progress(100, text="❌ 라벨 데이터 로드 실패")
                status_placeholder.error(f"❌ profile.txt 파일 로드 실패: {e}")
                time.sleep(1)
                progress_bar.empty()
                status_placeholder.empty()
        else:
            progress_bar.progress(100, text="시뮬레이션 라벨 데이터 생성")
            status_placeholder.warning("📁 profile.txt 파일을 찾을 수 없습니다. 시뮬레이션 데이터를 생성합니다.")
            time.sleep(1)
            progress_bar.empty()
            status_placeholder.empty()
    else:
        # 진행 상황 표시 없이 로드
        if os.path.exists(profile_path):
            try:
                labels = pd.read_csv(profile_path, sep='\t', header=None,
                                   names=['cooler', 'valve', 'pump', 'hydraulic', 'stable'], nrows=MAX_ROWS)
                return labels
            except Exception as e:
                st.warning(f"라벨 데이터 로드 실패: {e}")
    
    return generate_dummy_labels()

@st.cache_resource(ttl=3600)
def load_models(model_folder_path='models', show_progress=True):
    """저장된 모델 로드 (캐시됨)"""
    required_files = {
        'models.pkl': '머신러닝 모델',
        'scalers.pkl': '데이터 스케일러',
        'metadata.json': '모델 메타데이터'
    }
    
    if show_progress:
        progress_bar = st.progress(0, text="모델 파일을 확인하는 중입니다...")
        status_placeholder = st.empty()
        
        # 파일 존재 확인
        missing_files = []
        for filename, description in required_files.items():
            file_path = os.path.join(model_folder_path, filename)
            if not os.path.exists(file_path):
                missing_files.append(f"{filename} ({description})")
        
        if missing_files:
            progress_bar.progress(100, text="❌ 필요한 모델 파일이 없습니다")
            status_placeholder.error(f"❌ 다음 파일들을 찾을 수 없습니다: {', '.join(missing_files)}")
            time.sleep(2)
            progress_bar.empty()
            status_placeholder.empty()
            return None, None, None
        
        try:
            models, scalers, metadata = None, None, None
            
            # 모델 로드
            progress_bar.progress(30, text="🤖 머신러닝 모델 로드 중...")
            status_placeholder.info("🔄 models.pkl 파일을 읽고 있습니다...")
            with open(os.path.join(model_folder_path, 'models.pkl'), 'rb') as f:
                models = pickle.load(f)
            status_placeholder.success(f"✅ 머신러닝 모델 로드 완료! ({len(models)} 모델)")
            
            # 스케일러 로드
            progress_bar.progress(60, text="📏 데이터 스케일러 로드 중...")
            status_placeholder.info("🔄 scalers.pkl 파일을 읽고 있습니다...")
            with open(os.path.join(model_folder_path, 'scalers.pkl'), 'rb') as f:
                scalers = pickle.load(f)
            status_placeholder.success(f"✅ 데이터 스케일러 로드 완료! ({len(scalers)} 스케일러)")
            
            # 메타데이터 로드
            progress_bar.progress(90, text="📋 모델 메타데이터 로드 중...")
            status_placeholder.info("🔄 metadata.json 파일을 읽고 있습니다...")
            with open(os.path.join(model_folder_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            status_placeholder.success(f"✅ 모델 메타데이터 로드 완료!")
            
            progress_bar.progress(100, text="🎉 모든 모델 파일 로드 완료!")
            status_placeholder.success("🎉 모든 모델 파일이 성공적으로 로드되었습니다!")
            
            time.sleep(1)
            progress_bar.empty()
            status_placeholder.empty()
            
            return models, scalers, metadata
            
        except Exception as e:
            progress_bar.progress(100, text="❌ 모델 로드 중 오류 발생")
            status_placeholder.error(f"❌ 모델 로드 중 오류 발생: {e}")
            time.sleep(2)
            progress_bar.empty()
            status_placeholder.empty()
            return None, None, None
    else:
        # 진행 상황 표시 없이 로드
        try:
            models, scalers, metadata = None, None, None
            
            # 각 파일 로드
            for filename in required_files.keys():
                file_path = os.path.join(model_folder_path, filename)
                if not os.path.exists(file_path):
                    return None, None, None
            
            with open(os.path.join(model_folder_path, 'models.pkl'), 'rb') as f:
                models = pickle.load(f)
            with open(os.path.join(model_folder_path, 'scalers.pkl'), 'rb') as f:
                scalers = pickle.load(f)
            with open(os.path.join(model_folder_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return models, scalers, metadata
            
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None, None, None