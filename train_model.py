import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HydraulicSystemModelTrainer:
    """윈도우 기반 유압 시스템 이상 탐지 모델 훈련"""
    
    def __init__(self, data_path='data', window_seconds=10, stride_seconds=10):
        self.data_path = data_path
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        
        # 센서 정보 및 샘플링 레이트
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
        
        # 부품별 정상 조건
        self.components = {
            'cooler': {
                'name_kr': '냉각기',
                'normal_condition': lambda x: x >= 100,
                'label_col': 0,  # profile.txt의 컬럼 인덱스
                'states': {3: '거의 고장', 20: '효율 감소', 100: '정상'}
            },
            'valve': {
                'name_kr': '밸브',
                'normal_condition': lambda x: x >= 100,
                'label_col': 1,
                'states': {73: '거의 고장', 80: '심각한 지연', 90: '약간 지연', 100: '정상'}
            },
            'pump': {
                'name_kr': '펌프',
                'normal_condition': lambda x: x <= 0,
                'label_col': 2,
                'states': {0: '정상', 1: '약한 누출', 2: '심각한 누출'}
            },
            'hydraulic': {
                'name_kr': '유압',
                'normal_condition': lambda x: x >= 130,
                'label_col': 3,
                'states': {90: '거의 고장', 100: '심각히 낮음', 115: '약간 낮음', 130: '정상'}
            }
        }
        
    def load_data(self):
        """센서 데이터와 라벨 로드"""
        print("=== 데이터 로딩 중 ===")
        sensor_data = {}
        
        # 센서 데이터 로드
        for sensor, info in self.sensor_info.items():
            file_path = os.path.join(self.data_path, f'{sensor}.txt')
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, sep='\t', header=None)
                sensor_data[sensor] = data
                print(f"✓ {sensor} 로드 완료 ({info['freq']}Hz, {data.shape})")
        
        # 라벨 로드
        labels = pd.read_csv(
            os.path.join(self.data_path, 'profile.txt'), 
            sep='\t', 
            header=None,
            names=['cool', 'valve', 'pump', 'hydr', 'stable']
        )
        
        print(f"\n총 {len(labels)}개 사이클 로드 완료")
        
        # 부품별 상태 분포 출력
        print("\n=== 부품별 상태 분포 ===")
        for comp_name, comp_info in self.components.items():
            label_col = comp_info['label_col']
            print(f"\n{comp_info['name_kr']}:")
            counts = labels.iloc[:, label_col].value_counts().sort_index()
            for state, count in counts.items():
                meaning = comp_info['states'].get(state, str(state))
                print(f"  {state} ({meaning}): {count}개 ({count/len(labels)*100:.1f}%)")
        
        return sensor_data, labels
    
    def create_windows(self, sensor_data, labels):
        """60초 사이클을 10초 윈도우로 분할"""
        print(f"\n=== 윈도우 생성 중 ===")
        print(f"윈도우 크기: {self.window_seconds}초")
        print(f"이동 간격: {self.stride_seconds}초")
        
        window_data = []
        window_labels = []
        
        # 각 사이클 처리
        for cycle_idx in tqdm(range(len(labels)), desc="사이클 처리"):
            cycle_label = labels.iloc[cycle_idx]
            
            # 윈도우 시작 위치들 (0, 10, 20, 30, 40, 50초)
            window_starts = list(range(0, 60 - self.window_seconds + 1, self.stride_seconds))
            
            for window_start in window_starts:
                window_end = window_start + self.window_seconds
                window_dict = {}
                
                # 각 센서의 윈도우 데이터 추출
                for sensor_name, sensor_df in sensor_data.items():
                    freq = self.sensor_info[sensor_name]['freq']
                    
                    # 윈도우에 해당하는 샘플 인덱스
                    start_idx = int(window_start * freq)
                    end_idx = int(window_end * freq)
                    
                    # 해당 사이클의 센서 데이터에서 윈도우 추출
                    cycle_data = sensor_df.iloc[cycle_idx].values
                    window_samples = cycle_data[start_idx:end_idx]
                    
                    window_dict[sensor_name] = window_samples
                
                window_data.append(window_dict)
                window_labels.append(cycle_label)
        
        print(f"\n총 {len(window_data)}개 윈도우 생성 완료")
        print(f"(사이클당 {len(window_starts)}개 윈도우)")
        
        return window_data, window_labels
    
    def extract_features(self, window_data_list):
        """윈도우별 시간 도메인 특징 추출"""
        print("\n=== 시간 도메인 특징 추출 중 ===")
        
        all_features = []
        
        for window_data in tqdm(window_data_list, desc="특징 추출"):
            features = {}
            
            for sensor_name, sensor_values in window_data.items():
                values = np.array(sensor_values)
                
                if len(values) < 5:
                    continue
                
                # 기본 통계 특징
                features[f'{sensor_name}_mean'] = np.mean(values)
                features[f'{sensor_name}_std'] = np.std(values)
                features[f'{sensor_name}_max'] = np.max(values)
                features[f'{sensor_name}_min'] = np.min(values)
                features[f'{sensor_name}_rms'] = np.sqrt(np.mean(values**2))
                features[f'{sensor_name}_peak2peak'] = np.max(values) - np.min(values)
                
                # 고차 통계량
                features[f'{sensor_name}_skew'] = stats.skew(values)
                features[f'{sensor_name}_kurtosis'] = stats.kurtosis(values)
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        features_df = features_df.fillna(0)
        
        print(f"\n추출된 특징 shape: {features_df.shape}")
        print(f"특징 수: {len(features_df.columns)}")
        
        return features_df
    
    def train_models(self, features_df, window_labels):
        """부품별 OneClass SVM 모델 훈련"""
        print("\n=== 부품별 모델 훈련 ===")
        
        models = {}
        scalers = {}
        metadata = {
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'window_seconds': self.window_seconds,
            'stride_seconds': self.stride_seconds,
            'n_features': features_df.shape[1],
            'feature_columns': list(features_df.columns),
            'components': {}
        }
        
        # DataFrame으로 변환
        labels_df = pd.DataFrame(window_labels)
        
        for comp_name, comp_info in self.components.items():
            print(f"\n{comp_info['name_kr']} 모델 훈련 중...")
            
            # 해당 부품의 정상 여부 판단
            label_col = comp_info['label_col']
            normal_condition = comp_info['normal_condition']
            normal_mask = labels_df.iloc[:, label_col].apply(normal_condition)
            
            # 학습/검증 분할
            X_train, X_val, y_train, y_val = train_test_split(
                features_df, 
                ~normal_mask,  # 이상 라벨 (True: 이상, False: 정상)
                test_size=0.3, 
                random_state=42,
                stratify=~normal_mask
            )
            
            # 인덱스 리셋 (중요: 에러 해결)
            X_train = X_train.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)
            
            # 정상 데이터만으로 학습
            X_train_normal = X_train[~y_train]
            
            print(f"  학습 데이터 (정상만): {len(X_train_normal)}개")
            print(f"  검증 데이터: {len(X_val)}개 (정상: {sum(~y_val)}, 이상: {sum(y_val)})")
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_normal)
            X_val_scaled = scaler.transform(X_val)
            
            # OneClass SVM 훈련
            model = OneClassSVM(
                kernel='rbf', 
                gamma='scale', 
                nu=0.05  # 이상치 비율 추정
            )
            model.fit(X_train_scaled)
            
            # 검증 성능 평가
            val_pred = model.predict(X_val_scaled)
            val_pred_binary = (val_pred == -1)  # -1: 이상, 1: 정상
            
            # 성능 지표
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_val, val_pred_binary)
            precision = precision_score(y_val, val_pred_binary, zero_division=0)
            recall = recall_score(y_val, val_pred_binary, zero_division=0)
            f1 = f1_score(y_val, val_pred_binary, zero_division=0)
            
            print(f"  검증 성능:")
            print(f"    정확도: {accuracy:.3f}")
            print(f"    정밀도: {precision:.3f}")
            print(f"    재현율: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            
            # 저장
            models[comp_name] = model
            scalers[comp_name] = scaler
            metadata['components'][comp_name] = {
                'name_kr': comp_info['name_kr'],
                'n_normal_train': len(X_train_normal),
                'n_total_train': len(X_train),
                'validation_accuracy': float(accuracy),
                'validation_f1': float(f1)
            }
        
        return models, scalers, metadata
    
    def save_models(self, models, scalers, metadata, save_dir='models'):
        """모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델과 스케일러 저장
        with open(os.path.join(save_dir, 'models.pkl'), 'wb') as f:
            pickle.dump(models, f)
        
        with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)
        
        # 메타데이터 저장
        with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 모델이 {save_dir} 폴더에 저장되었습니다.")
        
        # 모델 요약 출력
        print("\n=== 모델 훈련 요약 ===")
        print(f"훈련 일시: {metadata['training_date']}")
        print(f"윈도우 크기: {metadata['window_seconds']}초")
        print(f"특징 수: {metadata['n_features']}")
        print("\n부품별 성능:")
        for comp_name, comp_data in metadata['components'].items():
            print(f"  {comp_data['name_kr']}: "
                  f"정확도={comp_data['validation_accuracy']:.3f}, "
                  f"F1={comp_data['validation_f1']:.3f}")

# 실행
if __name__ == "__main__":
    # 트레이너 초기화
    trainer = HydraulicSystemModelTrainer(
        data_path='data',
        window_seconds=10,
        stride_seconds=10
    )
    
    # 1. 데이터 로드
    sensor_data, labels = trainer.load_data()
    
    # 2. 윈도우 생성
    window_data, window_labels = trainer.create_windows(sensor_data, labels)
    
    # 3. 특징 추출
    features = trainer.extract_features(window_data)
    
    # 4. 모델 훈련
    models, scalers, metadata = trainer.train_models(features, window_labels)
    
    # 5. 모델 저장
    trainer.save_models(models, scalers, metadata)
    
    print("\n✅ 모델 훈련 및 저장 완료!")