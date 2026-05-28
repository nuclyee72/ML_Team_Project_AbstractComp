# 프로젝트 진행 계획: 인간 vs AI 생성 초록 비교 분석

현재 `proposal.txt`에 명시된 2. Method의 3) Data collect 단계까지 완료되었습니다. 수집된 데이터(`dbpia_computer_science.csv`)를 바탕으로 향후 진행할 데이터 분석 및 결과 도출 계획을 세분화하여 정리했습니다.

## 1. 데이터 전처리 및 준비 (Data Preprocessing)
- **데이터 로드 및 확인**: `pandas`를 이용하여 `dbpia_computer_science.csv` 파일 불러오기 및 결측치/이상치 점검.
- **텍스트 정제**: 
  - 특수문자, 불필요한 공백 제거.
  - `abstract`(인간 작성 초록)와 `fake_abstract`(AI 생성 초록) 컬럼 데이터 분리 및 구조화.

## 2. 지표 측정 및 특성 추출 (Feature Extraction)
`proposal.txt`의 분석 계획에 따라 데이터에서 필요한 수치를 추출합니다.

### 2.1. 구조적 무작위성 (Perplexity 측정)
- **방법**: 한국어 사전학습 언어 모델(예: KoGPT2 등)을 활용하여 각 텍스트의 Perplexity(당혹도)를 계산.
- **도구**: HuggingFace `transformers` 라이브러리 활용.

### 2.2. 언어적 복잡도 (Linguistic Complexity 측정)
- **방법**: 자연어 처리(NLP) 분석을 통해 문장의 정교함과 어휘 다양성 산출.
  - 평균 문장 길이 측정.
  - 어휘 다양성 지표(Type-Token Ratio, TTR) 계산.
  - 형태소 분석기(예: KoNLPy)를 활용한 품사(POS) 비율 및 구문적 복잡도 확인.

### 2.3. 초록 간 유사도 (Cosine Similarity 측정)
- **방법**: 텍스트 임베딩 모델(예: Sentence-BERT, KoBERT 등)로 텍스트를 벡터화한 후, 집단 간/집단 내 코사인 유사도(Cosine Similarity) 계산.
- **측정 대상**:
  - **그룹 내 유사도**: Human 초록 간의 유사도 평균 vs AI 초록 간의 유사도 평균.
  - **주제/문체 유사도**: 특정 Human 초록과 이를 모방한 AI 초록 간의 유사도 vs 완전히 다른 주제의 AI 초록 간의 유사도.

## 3. 통계 분석 및 가설 검증 (Statistical Analysis)
추출된 수치(Perplexity, 언어적 복잡도, 코사인 유사도)를 바탕으로 세웠던 4가지 가설을 검증합니다. Python의 `scipy.stats`를 활용할 예정입니다.

1. **구조적 무작위성 검증**: Perplexity 수치에 대한 독립표본 t-검정 수행 (가설: $Mean(Randomness_{Human}) > Mean(Randomness_{AI})$).
2. **언어적 복잡도 검증**: 복잡도 수치에 대한 독립표본 t-검정 수행 (가설: $Mean(Complexity_{Human}) < Mean(Complexity_{AI})$).
3. **그룹 내 유사도 검증**: Human 집단 내 코사인 유사도와 AI 집단 내 코사인 유사도의 평균 비교 (가설: $Mean(Cosine_{Human}) < Mean(Cosine_{AI})$).
4. **주제/문체 유사도 검증**: Human-AI(모방) 유사도와 AI-AI(무관) 유사도의 평균 비교 (가설: $Mean(Human \& AI) < Mean(AI \& AI)$).

## 4. 데이터 시각화 (Data Visualization)
- **분포 비교**: Perplexity와 언어적 복잡도에 대해 Human과 AI 집단의 차이를 보여주는 Boxplot 및 바이올린 플롯 시각화.
- **유사도 매트릭스**: 코사인 유사도 결과를 히트맵(Heatmap)이나 막대그래프로 시각화하여 그룹 간 차이를 명확히 표현.
- **도구**: `matplotlib`, `seaborn` 라이브러리 활용.

## 5. 최종 결과 도출 및 보고서 작성
- 통계 분석 결과를 종합하여 p-value 등 유의성 지표를 확인하고 가설의 채택/기각 여부 판별.
- AI 생성 텍스트의 구조적, 언어적 패턴 분석 결과를 바탕으로 "AI 텍스트 식별"에 대한 시사점 도출.
- 프로젝트 최종 결과 보고서(`Results` 파트) 작성.
