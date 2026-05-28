import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import joblib
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, PreTrainedTokenizerFast

st.set_page_config(page_title="AI 논문 초록 감별 대시보드", layout="wide")

st.title("🧠 AI 초록 감별기 (AI Abstract Detector) 데모 시스템")
st.markdown("""
본 시스템은 입력받은 컴퓨터공학/ICT 분야 논문 초록을 대상으로, **구조적 무작위성(PPL)**, **어휘 복잡도(TTR)**, 
그리고 **시맨틱 문체 임베딩**을 결합한 하이브리드 앙상블 모델을 활용해 AI 생성 여부를 고정밀도로 판별합니다.
""")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models():
    # 1. PPL 추출용 KoGPT2
    gpt_model_name = 'skt/kogpt2-base-v2'
    gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(gpt_model_name, bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
    gpt_model.eval()
    
    # 2. KR-SBERT 문장 임베딩 모델
    sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    
    # 3. PCA 모델 복원
    pca_model = joblib.load('pca_model.pkl')
    
    # 4. LightGBM 모델군 복원
    lgb_models = joblib.load('lgb_models.pkl')
    
    # 5. KoELECTRA Fine-tuned DL 모델
    electra_tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    electra_model = AutoModelForSequenceClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=2)
    if os.path.exists('best_koelectra_model.pt'):
        electra_model.load_state_dict(torch.load('best_koelectra_model.pt', map_location=device))
    electra_model.to(device)
    electra_model.eval()
    
    return gpt_tokenizer, gpt_model, sbert_model, pca_model, lgb_models, electra_tokenizer, electra_model

try:
    gpt_tokenizer, gpt_model, sbert_model, pca_model, lgb_models, electra_tokenizer, electra_model = load_models()
    st.success("감별 모델 복원 및 로드 완료!")
except Exception as e:
    st.error(f"모델 로드 중 에러 발생: {e}")

def calc_ppl(text):
    encodings = gpt_tokenizer(text, return_tensors='pt').to(device)
    max_length = gpt_model.config.n_positions
    stride = 512
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = gpt_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 감별할 논문 초록 입력")
    input_text = st.text_area("초록 원문을 여기에 붙여넣어 주세요.", height=300, placeholder="여기에 한국어 초록을 입력하고 Ctrl + Enter 또는 아래 감별 시작 버튼을 눌러주세요.")
    run_btn = st.button("🔍 초록 감별 및 분석 시작", use_container_width=True)

with col2:
    st.subheader("📊 실시간 분석 리포트")
    if run_btn and input_text.strip():
        with st.spinner("실시간 통계 및 딥러닝 특징 추출 중..."):
            # 1. 수치 지표 계산
            ppl = calc_ppl(input_text)
            words = input_text.split()
            ttr = len(set(words)) / len(words) if words else 0
            length = len(input_text)
            
            st.markdown(f"""
            - **구조적 무작위성 (PPL)**: `{ppl:.2f}` (평균: Human 73.1 / AI 54.3)
            - **어휘 다양성 (TTR)**: `{ttr:.4f}` (평균: Human 0.86 / AI 0.90)
            - **초록 글자 수**: `{length}` 자
            """)
            
            # 2. KoELECTRA 딥러닝 예측 확률
            inputs = electra_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=300).to(device)
            with torch.no_grad():
                outputs = electra_model(**inputs)
                dl_prob = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().item()
            
            # 3. SBERT 및 PCA 적용
            emb = sbert_model.encode([input_text])
            emb_pca = pca_model.transform(emb)[0]
            
            # LightGBM용 피처 매트릭스 구성
            ppl_clipped = min(ppl, 300)
            X_tab = np.hstack([np.array([ppl_clipped, ttr, length]), emb_pca]).reshape(1, -1)
            
            # 5개 LGBM 모델 예측 평균
            lgb_probs = []
            for model in lgb_models:
                lgb_probs.append(model.predict_proba(X_tab)[:, 1][0])
            lgb_prob_mean = np.mean(lgb_probs)
            
            # 최종 앙상블 결합 확률
            final_ai_prob = 0.4 * lgb_prob_mean + 0.6 * dl_prob
            
            st.divider()
            st.subheader("🎯 최종 감별 결과")
            if final_ai_prob >= 0.5:
                st.error(f"🚨 AI 생성물 감지! (AI 확률: **{final_ai_prob*100:.1f}%**)")
                st.info("💡 낮은 구조적 무작위성(PPL) 및 기계적으로 정형화된 어휘 사용 패턴이 감지되었습니다.")
            else:
                st.success(f"✅ 인간 작성물 판정! (AI 확률: **{final_ai_prob*100:.1f}%**)")
                st.info("💡 다양한 표현과 논리적 비기계적 문장 구조적 배치(높은 무작위성)가 보존되어 있습니다.")
    else:
        st.info("왼쪽 창에 텍스트를 입력하고 분석 버튼을 클릭하세요.")
