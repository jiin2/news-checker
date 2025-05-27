import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import google.generativeai as genai

# API 키는 .streamlit/secrets.toml에 저장하고 불러오기
openai.api_key = st.secrets["OPENAI_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 기사 본문 추출 함수 (newspaper3k 제거 → BeautifulSoup 대체)
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = '\n'.join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return article_text[:5000]  # 최대 길이 제한
    except Exception as e:
        return f"본문 추출 실패: {e}"

# 요약 함수 (Gemini)
def summarize_text(text):
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        system_instruction="너는 뉴스 기사의 핵심 내용을 객관적으로 요약하는 AI야."
    )
    prompt = f"""
    다음 뉴스 기사 본문을 객관적인 사실에 기반하여 핵심 내용 중심으로 요약해 주십시오.
    요약에는 주요 인물, 발생한 사건, 중요한 발언, 그리고 사건의 배경 정보가 포함되어야 합니다.
    주관적인 해석, 평가, 또는 기사에 명시적으로 드러나지 않은 추론은 배제하고, 사실 관계를 명확히 전달하는 데 집중해 주십시오.
    분량은 한국어 기준으로 약 3~5문장 (또는 100~150 단어) 정도로 간결하게 작성해 주십시오.

    기사:
    {text}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        )
        return response.text.strip()
    except Exception as e:
        st.warning("요약 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        return "요약 생성에 실패했습니다."

# 프레이밍 분석 함수 (GPT-4)
def detect_bias(title, text):
    prompt = f"""
    다음은 뉴스 제목과 본문입니다.
    제목이 본문 내용을 충분히 반영하고 있는지, 중요한 맥락이나 인물의 입장이 왜곡되거나 누락되었는지 판단해줘.

    제목: {title}
    본문: {text}

    분석 결과를 간단히 3~5줄로 정리해줘.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 공정한 뉴스 프레이밍 분석 도우미야."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

# GPT 기반 키워드 추출 함수
def extract_keywords_gpt(article_text):
    prompt = f"""
    다음 뉴스 기사 본문에서 가장 중요한 핵심 키워드를 5개만 추출하여, 각 키워드를 쉼표(,)로 구분한 하나의 문자열로 응답해줘. 다른 설명이나 문장은 포함하지 마.

    기사 본문:
    {article_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 뉴스 키워드 추출을 잘하는 요약봇이야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=100
    )
    keywords_string = response["choices"][0]["message"]["content"].strip()
    if ":" in keywords_string:
        keywords_string = keywords_string.split(":")[-1].strip()
    return [kw.strip() for kw in keywords_string.split(',') if kw.strip()]

# 유사도 측정 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit 인터페이스 시작
st.title("🧐 뉴스읽은척방지기")
st.write("기사 제목이 본문과 어울리는지, 왜곡됐는지 AI와 함께 분석해보자!")

url = st.text_input("뉴스 기사 URL을 입력하세요")

if st.button("검사 시작") and url:
    try:
        title = "기사 제목 추출 실패"
        text = get_article_text(url)

        body_summary = summarize_text(text)
        title_summary = summarize_text(title)

        embeddings = model.encode([title_summary, body_summary], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        if similarity > 0.75:
            result = "✅ 제목이 본문 내용을 잘 반영하고 있어요."
        elif similarity > 0.5:
            result = "🟡 제목이 본문과 약간 다를 수 있어요."
        else:
            result = "⚠️ 제목이 본문 내용과 많이 달라요. 낚시성일 수 있어요."

        extracted_keywords = extract_keywords_gpt(text)
        missing = [kw for kw in extracted_keywords if kw not in title]
        framing_result = detect_bias(title, text)

        st.subheader("📰 기사 제목")
        st.write(title)
        st.markdown(f"[기사 원문 바로가기]({url})")

        st.subheader("🧾 본문 요약")
        st.write(body_summary)
        with st.expander("⚠️ AI 요약에 대한 중요 안내 (클릭하여 확인)"):
            st.markdown("""
            - 본 요약은 Gemini 모델을 통해 생성되었습니다.
            - 모든 내용을 완벽히 반영하지 못할 수 있으며, 판단은 사용자에게 달려 있습니다.
            """)

        st.subheader("🔍 AI 추출 주요 키워드와 제목 비교")
        st.markdown(f"**본문 핵심 키워드:** {', '.join(extracted_keywords)}")
        if missing:
            st.warning(f"❗ 제목에서 다음 핵심 내용이 빠져 있어요: {', '.join(missing)}")
        else:
            st.success("✅ 제목에 핵심 키워드가 잘 반영되어 있어요.")

        st.subheader("📊 제목-본문 유사도 판단")
        st.write(result)

        st.subheader("🕵️ 프레이밍 분석 결과")
        with st.expander("⚠️ AI 프레이밍 분석 주의사항 (클릭하여 확인)"):
            st.markdown("""
            - 본 분석은 GPT 모델 기반이며, 완벽한 해석을 보장하지 않습니다.
            - 제공된 분석은 참고용이며 최종 판단은 사용자에게 있습니다.
            """)
        st.info(framing_result)

    except Exception as e:
        st.error(f"기사 처리 중 오류 발생: {str(e)}")
