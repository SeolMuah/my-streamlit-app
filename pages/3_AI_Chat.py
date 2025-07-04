import streamlit as st
import os
from google import genai
from google.genai import types

st.set_page_config(page_title="AI 채팅", page_icon="💬", layout="wide")

st.title("💬 AI 유지보수 도우미")
st.markdown("---")

# API 키 확인
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    st.warning("Gemini API 키가 설정되지 않았습니다.")
    st.info("""
    설정 방법:
    1. `.streamlit/secrets.toml` 파일 생성
    2. 다음 내용 추가:
    ```
    GEMINI_API_KEY = "your-api-key-here"
    ```
    """)
else:
    # Gemini 클라이언트 초기화
    client = genai.Client(api_key=api_key)
    
    # 세션 상태 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 채팅 설정
    generation_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=500,
        system_instruction="""너는 유압 시스템 전문가 AI 어시스턴트야. 
        사용자가 유압 시스템의 이상 징후, 유지보수, 문제 해결에 대해 질문하면 
        전문적이고 실용적인 조언을 제공해. 
        부품: 냉각기(Cooler), 밸브(Valve), 펌프(Pump), 유압(Hydraulic)
        센서: 압력(PS), 온도(TS), 유량(FS), 진동(VS), 전력(EPS)"""
    )
    
    # AI 응답 생성 함수 (중복 코드를 줄이기 위해 함수로 분리)
    def generate_ai_response():
        try:
            # 대화 히스토리를 Gemini 형식으로 변환
            messages = []
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                else:
                    messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
            
            # 응답 생성
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=messages,
                config=generation_config
            )
            
            # AI 응답 표시
            assistant_response = response.text
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            # 응답이 표시되도록 화면을 다시 그립니다.
            st.rerun() 
            
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {str(e)}")

    # 사이드바
    with st.sidebar:
        st.header("채팅 설정")
        
        # 예시 질문
        st.subheader("예시 질문")
        example_questions = [
            "펌프에서 이상이 감지되었습니다. 어떻게 해야 하나요?",
            "냉각기 효율이 떨어졌을 때 점검 사항은?",
            "압력 센서 값이 비정상적으로 높습니다.",
            "예방적 유지보수 일정은 어떻게 수립하나요?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"ex_{q}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                generate_ai_response() # 버튼 클릭 시 AI 응답 생성 함수 호출
        
        # 대화 초기화
        if st.button("🔄 대화 초기화"):
            st.session_state.chat_history = []
            st.rerun()
    
    # 채팅 인터페이스
    st.subheader("대화")
    
    # 대화 히스토리 표시
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # 입력창
    if prompt := st.chat_input("유압 시스템에 대해 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        generate_ai_response() # 입력창에 입력 시 AI 응답 생성 함수 호출