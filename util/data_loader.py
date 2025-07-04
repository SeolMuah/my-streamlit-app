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

# 센서별 주파수 정보
SENSOR_FREQUENCIES = {
    'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
    'EPS1': 100,
    'FS1': 10, 'FS2': 10,
    'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
    'VS1': 1,
    'CE': 1, 'CP': 1, 'SE': 1
}

def get_sensor_time_axis(sensor_name, cycle_duration=60):
    """센서별 시간축 생성 (60초 기준)"""
    freq = SENSOR_FREQUENCIES.get(sensor_name, 1)
    num_samples = freq * cycle_duration
    return np.linspace(0, cycle_duration, num_samples)

# HydraulicDataLoader 클래스
class HydraulicDataLoader:
    def __init__(self, data_folder_path, window_seconds=10, stride_seconds=10):
        self.data_folder_path = data_folder_path
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        
        # 센서 정보 (SENSOR_FREQUENCIES와 동일)
        self.sensor_info = {
            'PS1': {'freq': 100, 'type': 'pressure'},
            'PS2': {'freq': 100, 'type': 'pressure'},
            'PS3': {'freq': 100, 'type': 'pressure'},
            'PS4': {'freq': 100, 'type': 'pressure'},
            'PS5': {'freq': 100, 'type': 'pressure'},
            'PS6': {'freq': 100, 'type': 'pressure'},
            'EPS1': {'freq': 100, 'type': 'motor_power'},
            'FS1': {'freq': 10, 'type': 'flow'},
            'FS2': {'freq': 10, 'type': 'flow'},
            'TS1': {'freq': 1, 'type': 'temperature'},
            'TS2': {'freq': 1, 'type': 'temperature'},
            'TS3': {'freq': 1, 'type': 'temperature'},
            'TS4': {'freq': 1, 'type': 'temperature'},
            'VS1': {'freq': 1, 'type': 'vibration'},
            'CE': {'freq': 1, 'type': 'cooling_efficiency'},
            'CP': {'freq': 1, 'type': 'cooling_power'},
            'SE': {'freq': 1, 'type': 'efficiency_factor'}
        }
        
        self.sensor_data = {}
        self.profile_data = None
        self.current_cycle_index = 0
        self.num_cycles = 0

    def _load_all_data(self):
        """모든 센서 데이터와 프로파일 로드"""
        with st.spinner("데이터 로드 중..."):
            progress_text = "센서 데이터 파일 읽는 중. 잠시만 기다려 주세요."
            my_bar = st.progress(0, text=progress_text)
            
            if not os.path.exists(self.data_folder_path):
                st.error(f"경고: 데이터 폴더 '{self.data_folder_path}'를 찾을 수 없습니다. 시뮬레이션 데이터를 사용합니다.")
                self._generate_dummy_data()
                my_bar.empty()
                return
            
            try:
                # 센서 데이터 로드
                all_files_found = True
                sensor_names = list(self.sensor_info.keys())
                
                for i, sensor_name in enumerate(sensor_names):
                    file_path = os.path.join(self.data_folder_path, f"{sensor_name}.txt")
                    
                    my_bar.progress(int(((i + 1) / len(sensor_names)) * 80), 
                                   text=f"{sensor_name}.txt 로드 중...")
                    
                    if os.path.exists(file_path):
                        data = pd.read_csv(file_path, sep='\t', header=None)
                        self.sensor_data[sensor_name] = data
                    else:
                        st.warning(f"센서 '{sensor_name}' 파일을 찾을 수 없습니다.")
                        all_files_found = False
                        break
                
                # 프로파일 데이터 로드
                profile_path = os.path.join(self.data_folder_path, 'profile.txt')
                if os.path.exists(profile_path):
                    my_bar.progress(90, text="프로파일 데이터 로드 중...")
                    self.profile_data = pd.read_csv(
                        profile_path, 
                        sep='\t', 
                        header=None,
                        names=['cooler', 'valve', 'pump', 'hydraulic', 'stable']
                    )
                    self.num_cycles = len(self.profile_data)
                    my_bar.progress(100, text="데이터 로드 완료.")
                else:
                    st.warning("프로파일 데이터를 찾을 수 없습니다.")
                    all_files_found = False
                
                my_bar.empty()
                
                if not all_files_found or len(self.sensor_data) == 0:
                    self._generate_dummy_data()
                    st.warning("실제 데이터를 로드할 수 없어 시뮬레이션 데이터를 사용합니다.")
                else:
                    st.success(f"데이터 로드 완료! 총 {self.num_cycles}개 사이클")
                    
            except Exception as e:
                st.error(f"데이터 로드 중 오류 발생: {e}")
                my_bar.empty()
                self._generate_dummy_data()

    def _generate_dummy_data(self):
        """더미 데이터 생성"""
        self.num_cycles = 100
        
        # 각 센서별 더미 데이터 생성
        for sensor_name, info in self.sensor_info.items():
            freq = info['freq']
            cycle_length = freq * 60  # 60초 사이클
            
            # 센서 타입별 기본값 설정
            if sensor_name.startswith('PS'):
                base_value = np.random.uniform(140, 160)
            elif sensor_name.startswith('TS'):
                base_value = np.random.uniform(35, 40)
            elif sensor_name.startswith('FS'):
                base_value = np.random.uniform(8, 12)
            elif sensor_name == 'EPS1':
                base_value = np.random.uniform(2000, 2500)
            elif sensor_name == 'VS1':
                base_value = np.random.uniform(0.5, 1.0)
            else:  # CE, CP, SE
                base_value = np.random.uniform(0.8, 1.2)
            
            # 사이클별 데이터 생성
            dummy_data = []
            for cycle in range(self.num_cycles):
                cycle_data = base_value + np.random.normal(0, base_value * 0.03, cycle_length)
                dummy_data.append(cycle_data)
            
            self.sensor_data[sensor_name] = pd.DataFrame(dummy_data)
        
        # 더미 프로파일 데이터 생성
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
        
        if self.current_cycle_index >= self.num_cycles:
            self.current_cycle_index = 0
        
        # 현재 사이클에서 랜덤하게 윈도우 시작 위치 선택
        cycle_idx = self.current_cycle_index
        window_start = np.random.choice([0, 10, 20, 30, 40, 50])  # 0~50초 중 하나
        window_end = window_start + self.window_seconds
        
        window_data = {}
        
        # 각 센서의 윈도우 데이터 추출
        for sensor_name, sensor_df in self.sensor_data.items():
            freq = self.sensor_info[sensor_name]['freq']
            
            start_idx = int(window_start * freq)
            end_idx = int(window_end * freq)
            
            # 해당 사이클의 센서 데이터
            cycle_data = sensor_df.iloc[cycle_idx].values
            window_samples = cycle_data[start_idx:end_idx]
            
            window_data[sensor_name] = window_samples
        
        # 해당 사이클의 라벨 정보
        cycle_labels = self.profile_data.iloc[cycle_idx] if self.profile_data is not None else None
        
        self.current_cycle_index += 1
        
# 특징 추출 함수
def extract_features_from_window(window_data):
    """윈도우 데이터에서 특징 추출 (NaN 처리 강화)"""
    features = {}
    
    for sensor_name, sensor_values in window_data.items():
        values = np.array(sensor_values)
        
        # 유효한 값만 선택 (NaN, inf 제거)
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) < 2:
            # 유효한 값이 부족한 경우 기본값 사용
            features[f'{sensor_name}_mean'] = 0.0
            features[f'{sensor_name}_std'] = 0.0
            features[f'{sensor_name}_max'] = 0.0
            features[f'{sensor_name}_min'] = 0.0
            features[f'{sensor_name}_rms'] = 0.0
            features[f'{sensor_name}_peak2peak'] = 0.0
            features[f'{sensor_name}_skew'] = 0.0
            features[f'{sensor_name}_kurtosis'] = 0.0
            continue
        
        # 기본 통계 특징
        try:
            features[f'{sensor_name}_mean'] = np.mean(valid_values)
        except:
            features[f'{sensor_name}_mean'] = 0.0
            
        try:
            features[f'{sensor_name}_std'] = np.std(valid_values)
        except:
            features[f'{sensor_name}_std'] = 0.0
            
        try:
            features[f'{sensor_name}_max'] = np.max(valid_values)
        except:
            features[f'{sensor_name}_max'] = 0.0
            
        try:
            features[f'{sensor_name}_min'] = np.min(valid_values)
        except:
            features[f'{sensor_name}_min'] = 0.0
            
        try:
            features[f'{sensor_name}_rms'] = np.sqrt(np.mean(valid_values**2))
        except:
            features[f'{sensor_name}_rms'] = 0.0
            
        try:
            features[f'{sensor_name}_peak2peak'] = np.max(valid_values) - np.min(valid_values)
        except:
            features[f'{sensor_name}_peak2peak'] = 0.0
            
        # 고차 통계 특징
        try:
            if len(valid_values) > 2:
                from scipy.stats import skew, kurtosis
                skew_val = skew(valid_values)
                features[f'{sensor_name}_skew'] = skew_val if np.isfinite(skew_val) else 0.0
            else:
                features[f'{sensor_name}_skew'] = 0.0
        except:
            features[f'{sensor_name}_skew'] = 0.0
            
        try:
            if len(valid_values) > 2:
                from scipy.stats import skew, kurtosis
                kurt_val = kurtosis(valid_values)
                features[f'{sensor_name}_kurtosis'] = kurt_val if np.isfinite(kurt_val) else 0.0
            else:
                features[f'{sensor_name}_kurtosis'] = 0.0
        except:
            features[f'{sensor_name}_kurtosis'] = 0.0
    
    return features

# 이상 탐지 함수
def detect_anomaly(features, models, scalers, metadata):
    """각 부품별 이상 탐지 (NaN 처리 강화)"""
    # DataFrame 생성
    features_df = pd.DataFrame([features])
    
    # metadata에서 feature_columns 가져오기
    feature_cols = metadata.get('feature_columns', [])
    
    # 필요한 특징만 선택하고 누락된 특징은 0으로 채움
    features_selected = features_df.reindex(columns=feature_cols, fill_value=0.0)
    
    # NaN 값을 0으로 대체
    features_selected = features_selected.fillna(0.0)
    
    # inf 값도 처리
    features_selected = features_selected.replace([np.inf, -np.inf], 0.0)
    
    # 최종 검증: 여전히 NaN이 있는지 확인
    if features_selected.isnull().any().any():
        st.warning("특징 데이터에 여전히 NaN 값이 있습니다. 모든 NaN을 0으로 대체합니다.")
        features_selected = features_selected.fillna(0.0)
    
    results = {}
    
    for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
        try:
            # 스케일링 전 데이터 검증
            X_input = features_selected.copy()
            
            # 스케일링
            X_scaled = scalers[comp_name].transform(X_input)
            
            # 스케일링 후에도 NaN 체크
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                st.warning(f"{comp_name} 스케일링 후 NaN/inf 값 발견. 0으로 대체합니다.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 예측
            prediction = models[comp_name].predict(X_scaled)[0]
            score = models[comp_name].decision_function(X_scaled)[0]
            
            # 점수가 NaN인 경우 처리
            if np.isnan(score) or np.isinf(score):
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

def generate_dummy_sensor_data():
    """더미 센서 데이터 생성"""
    sensor_list = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1']
    sensor_data = {}
    
    for sensor in sensor_list:
        # 100개 사이클, 각 사이클당 주파수에 따른 데이터 포인트
        num_cycles = 100
        freq = SENSOR_FREQUENCIES.get(sensor, 100)
        cycle_length = freq * 60  # 60초 사이클
        
        # 센서 타입별 기본값 설정
        if sensor.startswith('PS'):
            base_value = np.random.uniform(140, 160)
        elif sensor == 'EPS1':
            base_value = np.random.uniform(2000, 2500)
        else:
            base_value = np.random.uniform(35, 40)
        
        # 사이클별 데이터 생성
        dummy_data = []
        for cycle in range(num_cycles):
            cycle_data = base_value + np.random.normal(0, base_value * 0.03, cycle_length)
            dummy_data.append(cycle_data)
        
        sensor_data[sensor] = pd.DataFrame(dummy_data)
    
    return sensor_data

def generate_dummy_labels():
    """더미 라벨 데이터 생성"""
    num_cycles = 100
    
    labels = pd.DataFrame({
        'cooler': np.random.choice([3, 20, 100], num_cycles, p=[0.1, 0.2, 0.7]),
        'valve': np.random.choice([73, 80, 90, 100], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'pump': np.random.choice([0, 1, 2], num_cycles, p=[0.7, 0.2, 0.1]),
        'hydraulic': np.random.choice([90, 100, 115, 130], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'stable': np.ones(num_cycles, dtype=int)
    })
    
    return labels

# 데이터 로더 인스턴스 생성 (캐시됨)
@st.cache_resource(ttl=3600)
def get_data_loader(data_folder_path='data'):
    """데이터 로더 인스턴스 생성 (캐시됨)"""
    loader = HydraulicDataLoader(data_folder_path)
    loader._load_all_data()
    return loader

# 센서별 주파수 정보
SENSOR_FREQUENCIES = {
    'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100,
    'EPS1': 100,
    'FS1': 10, 'FS2': 10,
    'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1,
    'VS1': 1,
    'CE': 1, 'CP': 1, 'SE': 1
}

def get_sensor_time_axis(sensor_name, cycle_duration=60):
    """센서별 시간축 생성 (60초 기준)"""
    freq = SENSOR_FREQUENCIES.get(sensor_name, 1)
    num_samples = freq * cycle_duration
    return np.linspace(0, cycle_duration, num_samples)
    

# 캐시된 데이터 로더 함수들
@st.cache_data(ttl=3600)  # 1시간 동안 캐시
def load_sensor_data(data_folder_path='data', show_progress=True):
    """센서 데이터 로드 (캐시됨)"""
    sensor_list = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 
                   'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 
                   'CE', 'CP', 'SE']
    
    sensor_data = {}
    
    if show_progress:
        progress_text = "센서 데이터 파일을 확인하는 중입니다..."
        progress_bar = st.progress(0, text=progress_text)
        status_placeholder = st.empty()
        
        total_sensors = len(sensor_list)
        loaded_count = 0
        
        # 파일 존재 확인
        existing_files = []
        for sensor in sensor_list:
            file_path = os.path.join(data_folder_path, f'{sensor}.txt')
            if os.path.exists(file_path):
                existing_files.append(sensor)
        
        if not existing_files:
            progress_bar.progress(100, text="실제 파일이 없어 시뮬레이션 데이터를 생성합니다...")
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
                with status_placeholder.container():
                    st.info(f"🔄 {sensor}.txt 파일을 읽고 있습니다...")
                
                data = pd.read_csv(file_path, sep='\t', header=None)
                sensor_data[sensor] = data
                loaded_count += 1
                
                with status_placeholder.container():
                    st.success(f"✅ {sensor}.txt 파일 로드 완료! ({len(data)} 행)")
                
                time.sleep(0.1)  # 사용자가 볼 수 있도록 잠시 대기
                
            except Exception as e:
                with status_placeholder.container():
                    st.error(f"❌ {sensor}.txt 파일 로드 실패: {e}")
                time.sleep(0.5)
        
        progress_bar.progress(100, text=f"✅ 센서 데이터 로딩 완료! ({loaded_count}/{len(existing_files)} 파일)")
        
        if loaded_count == 0:
            status_placeholder.warning("실제 센서 데이터를 찾을 수 없습니다. 시뮬레이션 데이터를 생성합니다.")
            sensor_data = generate_dummy_sensor_data()
        else:
            status_placeholder.success(f"🎉 총 {loaded_count}개의 센서 데이터 파일이 성공적으로 로드되었습니다!")
        
        time.sleep(1)
        progress_bar.empty()
        status_placeholder.empty()
        
    else:
        # 진행 상황 표시 없이 로드 (기존 방식)
        for sensor in sensor_list:
            file_path = os.path.join(data_folder_path, f'{sensor}.txt')
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path, sep='\t', header=None)
                    sensor_data[sensor] = data
                except Exception as e:
                    st.warning(f"센서 {sensor} 데이터 로드 실패: {e}")
        
        if not sensor_data:
            sensor_data = generate_dummy_sensor_data()
    
    return sensor_data

@st.cache_data(ttl=3600)  # 1시간 동안 캐시
def load_labels(data_folder_path='data', show_progress=True):
    """라벨 데이터 로드 (캐시됨)"""
    profile_path = os.path.join(data_folder_path, 'profile.txt')
    
    if show_progress:
        progress_bar = st.progress(0, text="라벨 데이터 파일을 확인하는 중입니다...")
        status_placeholder = st.empty()
        
        if os.path.exists(profile_path):
            try:
                progress_bar.progress(25, text="📋 profile.txt 파일 확인 완료")
                status_placeholder.info("🔄 profile.txt 파일을 읽고 있습니다...")
                time.sleep(0.2)
                
                progress_bar.progress(50, text="📋 profile.txt 파일 읽는 중...")
                labels = pd.read_csv(profile_path, sep='\t', header=None,
                                   names=['cooler', 'valve', 'pump', 'hydraulic', 'stable'])
                
                progress_bar.progress(75, text="📋 라벨 데이터 검증 중...")
                status_placeholder.success(f"✅ profile.txt 파일 로드 완료! ({len(labels)} 라벨)")
                time.sleep(0.2)
                
                progress_bar.progress(100, text="✅ 라벨 데이터 로딩 완료!")
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
            progress_bar.progress(100, text="❌ 라벨 파일이 없어 시뮬레이션 데이터 생성")
            status_placeholder.warning("📁 profile.txt 파일을 찾을 수 없습니다. 시뮬레이션 데이터를 생성합니다.")
            time.sleep(1)
            
            progress_bar.empty()
            status_placeholder.empty()
    else:
        # 진행 상황 표시 없이 로드
        if os.path.exists(profile_path):
            try:
                labels = pd.read_csv(profile_path, sep='\t', header=None,
                                   names=['cooler', 'valve', 'pump', 'hydraulic', 'stable'])
                return labels
            except Exception as e:
                st.warning(f"라벨 데이터 로드 실패: {e}")
    
    # 실패 시 더미 데이터 생성
    return generate_dummy_labels()

@st.cache_resource(ttl=3600)  # 1시간 동안 캐시
def load_models(model_folder_path='models', show_progress=True):
    """저장된 모델 로드 (캐시됨)"""
    models = None
    scalers = None
    metadata = None
    
    if show_progress:
        progress_text = "모델 파일을 확인하는 중입니다..."
        progress_bar = st.progress(0, text=progress_text)
        status_placeholder = st.empty()
        
        # 필요한 파일들
        required_files = {
            'models.pkl': '머신러닝 모델',
            'scalers.pkl': '데이터 스케일러',
            'metadata.json': '모델 메타데이터'
        }
        
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
            # models.pkl 로드
            progress_bar.progress(10, text="🤖 머신러닝 모델 파일 로드 중...")
            status_placeholder.info("🔄 models.pkl 파일을 읽고 있습니다...")
            time.sleep(0.2)
            
            model_path = os.path.join(model_folder_path, 'models.pkl')
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
            
            progress_bar.progress(40, text="✅ 머신러닝 모델 로드 완료")
            status_placeholder.success(f"✅ 머신러닝 모델 로드 완료! ({len(models)} 모델)")
            time.sleep(0.3)

            # scalers.pkl 로드
            progress_bar.progress(50, text="📏 데이터 스케일러 파일 로드 중...")
            status_placeholder.info("🔄 scalers.pkl 파일을 읽고 있습니다...")
            time.sleep(0.2)
            
            scaler_path = os.path.join(model_folder_path, 'scalers.pkl')
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            
            progress_bar.progress(70, text="✅ 데이터 스케일러 로드 완료")
            status_placeholder.success(f"✅ 데이터 스케일러 로드 완료! ({len(scalers)} 스케일러)")
            time.sleep(0.3)

            # metadata.json 로드
            progress_bar.progress(80, text="📋 모델 메타데이터 파일 로드 중...")
            status_placeholder.info("🔄 metadata.json 파일을 읽고 있습니다...")
            time.sleep(0.2)
            
            metadata_path = os.path.join(model_folder_path, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            progress_bar.progress(90, text="✅ 모델 메타데이터 로드 완료")
            status_placeholder.success(f"✅ 모델 메타데이터 로드 완료! ({len(metadata)} 항목)")
            time.sleep(0.3)

            progress_bar.progress(100, text="🎉 모든 모델 파일 로드 완료!")
            status_placeholder.success("🎉 모든 모델 파일이 성공적으로 로드되었습니다!")
            time.sleep(1)
            
            progress_bar.empty()
            status_placeholder.empty()
            
            return models, scalers, metadata
            
        except FileNotFoundError as e:
            progress_bar.progress(100, text="❌ 모델 파일을 찾을 수 없습니다")
            status_placeholder.error(f"❌ 필요한 모델 파일이 없습니다: {e}")
            status_placeholder.info("💡 train_model.py를 실행하여 모델을 훈련하고 저장하세요.")
            time.sleep(2)
            progress_bar.empty()
            status_placeholder.empty()
            return None, None, None
        except Exception as e:
            progress_bar.progress(100, text="❌ 모델 로드 중 오류 발생")
            status_placeholder.error(f"❌ 모델 로드 중 오류 발생: {e}")
            time.sleep(2)
            progress_bar.empty()
            status_placeholder.empty()
            return None, None, None
    else:
        # 진행 상황 표시 없이 로드 (기존 방식)
        try:
            # models.pkl 로드
            model_path = os.path.join(model_folder_path, 'models.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
            else:
                return None, None, None

            # scalers.pkl 로드
            scaler_path = os.path.join(model_folder_path, 'scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
            else:
                return None, None, None

            # metadata.json 로드
            metadata_path = os.path.join(model_folder_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            return models, scalers, metadata
            
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None, None, None

def generate_dummy_sensor_data():
    """더미 센서 데이터 생성"""
    sensor_list = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1']
    sensor_data = {}
    
    for sensor in sensor_list:
        # 100개 사이클, 각 사이클당 6000개 데이터 포인트 (60초 * 100Hz)
        num_cycles = 100
        cycle_length = 6000
        
        # 센서 타입별 기본값 설정
        if sensor.startswith('PS'):
            base_value = np.random.uniform(140, 160)
        elif sensor == 'EPS1':
            base_value = np.random.uniform(2000, 2500)
        else:
            base_value = np.random.uniform(35, 40)
        
        # 사이클별 데이터 생성
        dummy_data = []
        for cycle in range(num_cycles):
            cycle_data = base_value + np.random.normal(0, base_value * 0.03, cycle_length)
            dummy_data.append(cycle_data)
        
        sensor_data[sensor] = pd.DataFrame(dummy_data)
    
    return sensor_data

def generate_dummy_labels():
    """더미 라벨 데이터 생성"""
    num_cycles = 100
    
    labels = pd.DataFrame({
        'cooler': np.random.choice([3, 20, 100], num_cycles, p=[0.1, 0.2, 0.7]),
        'valve': np.random.choice([73, 80, 90, 100], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'pump': np.random.choice([0, 1, 2], num_cycles, p=[0.7, 0.2, 0.1]),
        'hydraulic': np.random.choice([90, 100, 115, 130], num_cycles, p=[0.1, 0.1, 0.2, 0.6]),
        'stable': np.ones(num_cycles, dtype=int)
    })
    
    return labels

# 특징 추출 함수
def extract_features_from_window(window_data):
    """윈도우 데이터에서 특징 추출 (NaN 처리 강화)"""
    features = {}
    
    for sensor_name, sensor_values in window_data.items():
        values = np.array(sensor_values)
        
        # 유효한 값만 선택 (NaN, inf 제거)
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) < 2:
            # 유효한 값이 부족한 경우 기본값 사용
            features[f'{sensor_name}_mean'] = 0.0
            features[f'{sensor_name}_std'] = 0.0
            features[f'{sensor_name}_max'] = 0.0
            features[f'{sensor_name}_min'] = 0.0
            features[f'{sensor_name}_rms'] = 0.0
            features[f'{sensor_name}_peak2peak'] = 0.0
            features[f'{sensor_name}_skew'] = 0.0
            features[f'{sensor_name}_kurtosis'] = 0.0
            continue
        
        # 기본 통계 특징
        try:
            features[f'{sensor_name}_mean'] = np.mean(valid_values)
        except:
            features[f'{sensor_name}_mean'] = 0.0
            
        try:
            features[f'{sensor_name}_std'] = np.std(valid_values)
        except:
            features[f'{sensor_name}_std'] = 0.0
            
        try:
            features[f'{sensor_name}_max'] = np.max(valid_values)
        except:
            features[f'{sensor_name}_max'] = 0.0
            
        try:
            features[f'{sensor_name}_min'] = np.min(valid_values)
        except:
            features[f'{sensor_name}_min'] = 0.0
            
        try:
            features[f'{sensor_name}_rms'] = np.sqrt(np.mean(valid_values**2))
        except:
            features[f'{sensor_name}_rms'] = 0.0
            
        try:
            features[f'{sensor_name}_peak2peak'] = np.max(valid_values) - np.min(valid_values)
        except:
            features[f'{sensor_name}_peak2peak'] = 0.0
            
        # 고차 통계 특징
        try:
            if len(valid_values) > 2:
                skew_val = skew(valid_values)
                features[f'{sensor_name}_skew'] = skew_val if np.isfinite(skew_val) else 0.0
            else:
                features[f'{sensor_name}_skew'] = 0.0
        except:
            features[f'{sensor_name}_skew'] = 0.0
            
        try:
            if len(valid_values) > 2:
                kurt_val = kurtosis(valid_values)
                features[f'{sensor_name}_kurtosis'] = kurt_val if np.isfinite(kurt_val) else 0.0
            else:
                features[f'{sensor_name}_kurtosis'] = 0.0
        except:
            features[f'{sensor_name}_kurtosis'] = 0.0
    
    return features

# 이상 탐지 함수
def detect_anomaly(features, models, scalers, metadata):
    """각 부품별 이상 탐지 (NaN 처리 강화)"""
    # DataFrame 생성
    features_df = pd.DataFrame([features])
    
    # metadata에서 feature_columns 가져오기
    feature_cols = metadata.get('feature_columns', [])
    
    # 필요한 특징만 선택하고 누락된 특징은 0으로 채움
    features_selected = features_df.reindex(columns=feature_cols, fill_value=0.0)
    
    # NaN 값을 0으로 대체
    features_selected = features_selected.fillna(0.0)
    
    # inf 값도 처리
    features_selected = features_selected.replace([np.inf, -np.inf], 0.0)
    
    # 최종 검증: 여전히 NaN이 있는지 확인
    if features_selected.isnull().any().any():
        st.warning("특징 데이터에 여전히 NaN 값이 있습니다. 모든 NaN을 0으로 대체합니다.")
        features_selected = features_selected.fillna(0.0)
    
    results = {}
    
    for comp_name in ['cooler', 'valve', 'pump', 'hydraulic']:
        try:
            # 스케일링 전 데이터 검증
            X_input = features_selected.copy()
            
            # 스케일링
            X_scaled = scalers[comp_name].transform(X_input)
            
            # 스케일링 후에도 NaN 체크
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                st.warning(f"{comp_name} 스케일링 후 NaN/inf 값 발견. 0으로 대체합니다.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 예측
            prediction = models[comp_name].predict(X_scaled)[0]
            score = models[comp_name].decision_function(X_scaled)[0]
            
            # 점수가 NaN인 경우 처리
            if np.isnan(score) or np.isinf(score):
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

# 데이터 로더 인스턴스 생성 (캐시됨)
@st.cache_resource(ttl=3600)
def get_data_loader(data_folder_path='data'):
    """데이터 로더 인스턴스 생성 (캐시됨)"""
    loader = HydraulicDataLoader(data_folder_path)
    loader._load_all_data()
    return loader