# 📊 소득 예측 모델 (Income Prediction Model)

## 📌 프로젝트 개요
이 프로젝트는 **미국 청소년 종단 연구(NLSY97)** 데이터를 활용하여 **개인의 소득(EARNINGS)**을 예측하는 **다변량 회귀 모델**을 구축하는 것이 목표입니다.  
이를 통해 학력, 근무 경험, 기타 요인이 소득에 미치는 영향을 분석하고 **데이터 기반의 정책적 인사이트**를 제공합니다.

---

## 📂 데이터 개요
사용된 데이터셋: **NLSY97_subset.csv**  
해당 데이터는 1997년 이후 미국 청소년 패널 데이터를 기반으로, 교육, 경력, 소득 등의 정보를 포함하고 있습니다.

### 주요 변수:
- `EARNINGS` - 시간당 소득 (2011년 인터뷰 기준)
- `S` - 학력 연수 (최종 학년 기준)
- `EXP` - 총 근무 경력 (학교 졸업 후 근무한 총 연수)
- `MARRIED` - 결혼 여부 (0=미혼, 1=기혼)
- `HOURS` - 주당 평균 근무 시간
- `TENURE` - 현재 직장에서의 근속 연수
- `ETHWHITE`, `ETHBLACK`, `ETHHISP` - 인종 정보

---

## ⚙️ 분석 방법

### 1️⃣ 데이터 전처리
- 결측값 처리 (중앙값 대체)
- 중복 데이터 제거
- 이상치 탐색 및 제거
- 범주형 변수 (성별, 인종) 원-핫 인코딩

### 2️⃣ 단순 선형 회귀 (Simple Linear Regression)
- **목표:** 학력(`S`)이 소득(`EARNINGS`)에 미치는 영향 분석
- **모델 학습:**
  ```python
  X_simple = df[['S']]
  y = df['EARNINGS']
  model_simple = LinearRegression()
  model_simple.fit(X_simple, y)
  ```
- **결과:**
  - Training R²: 0.0831
  - Testing R²: 0.0610
  - Training MSE: 134.8307
  - Testing MSE: 122.8715

### 3️⃣ 다변량 회귀 (Multivariable Regression)
- **목표:** 학력(`S`)과 경력(`EXP`)을 동시에 사용하여 소득 예측
- **모델 학습:**
  ```python
  X_multi = df[['S', 'EXP']]
  model_multi = LinearRegression()
  model_multi.fit(X_multi, y)
  ```
- **결과:**
  - Training R²: 0.1195
  - Testing R²: 0.0731
  - Training MSE: 129.4868
  - Testing MSE: 121.2776

### 4️⃣ 다항 회귀 (Polynomial Regression)
- **목표:** 변수 간의 비선형 관계를 반영하여 예측 성능 개선
- **추가 변수:** `HHINC97`(가구 소득), `HOURS`(근무 시간), `EDUCBA`(학사 학위 여부) 등 포함
- **다항 변환 적용:**
  ```python
  poly = PolynomialFeatures(degree=2, include_bias=False)
  X_train_poly = poly.fit_transform(X_train_scaled)
  X_test_poly = poly.transform(X_test_scaled)
  ```
- **결과:**
  - Training R²: 0.2050
  - Testing R²: 0.1354
  - Training MSE: 116.9079
  - Testing MSE: 113.1257
  - Training MAE: 6.8688
  - Testing MAE: 7.0137

### 5️⃣ 모델 평가
- **잔차 분석:**
  - Training Residual Mean: 0.0000
  - Testing Residual Mean: 0.1558
  - 잔차 분포가 정규성을 따르지 않고, 특정 패턴을 보임 → 추가적인 모델 개선 필요
- **모델 계수 해석:**
  ```python
  coefficients = dict(zip(poly.get_feature_names_out(features), model.coef_))
  ```
  - `S`(학력)의 회귀 계수: **4.1408** → 학력 1년 증가 시 예상 소득 약 4.14 증가
  - `EXP`(근무 경력)의 회귀 계수: **2.7608** → 근무 연수 1년 증가 시 예상 소득 약 2.76 증가
  - `FEMALE`(여성 변수): **-0.8689** → 여성인 경우 예상 소득이 약 0.87 낮음
  - `HOURS`(근무 시간): **0.7907** → 근무 시간이 길수록 소득 증가 경향

---

## 🔍 결론 및 개선 방향
**모델 성능 개선**
1. **비선형 회귀 모델 적용**: 다항 회귀를 추가했으나, 추가적인 비선형 변환이나 랜덤 포레스트, XGBoost 등의 알고리즘 고려
2. **추가적인 변수를 활용**: `TENURE`, `JOB CATEGORY` 등 직업 관련 변수 추가
3. **데이터 정규성 검토**: 이상치 처리 및 로그 변환 적용 가능성 고려

**향후 계획**
- 다른 머신러닝 모델 (랜덤 포레스트, XGBoost 등)과 비교 실험
- 모델 해석력 향상을 위한 피처 엔지니어링
- 실제 정책 수립에 활용 가능한 데이터 인사이트 제공

---

### 작성자
- **프로젝트 담당:** 안제호
- **문의:** [ajh4234@gmail.com](mailto:ajh4234@gmail.com)

