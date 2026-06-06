import streamlit as st
import joblib
import numpy as np
from deep_translator import GoogleTranslator

model = joblib.load("job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 1. 영문 직무 -> 한글 매핑 사전
job_korean = {
    "ENGINEERING": "엔지니어링", "INFORMATION-TECHNOLOGY": "정보기술(IT)",
    "BUSINESS-DEVELOPMENT": "사업개발", "HR": "인사(HR)", "ACCOUNTANT": "회계",
    "FINANCE": "금융", "HEALTHCARE": "보건·의료", "SALES": "영업",
    "TEACHER": "교육", "DESIGNER": "디자인", "DIGITAL-MEDIA": "디지털미디어",
    "PUBLIC-RELATIONS": "홍보(PR)", "CONSULTANT": "컨설팅", "BANKING": "은행",
    "CONSTRUCTION": "건설", "AGRICULTURE": "농업", "AUTOMOBILE": "자동차",
    "CHEF": "조리·외식", "FITNESS": "스포츠·피트니스", "AVIATION": "항공",
    "ADVOCATE": "법률", "APPAREL": "패션·의류", "ARTS": "예술", "BPO": "고객서비스(BPO)"
}

# 2. 직무별 추천 자격증 매핑 사전
job_certificates = {
    "ENGINEERING": "기사 자격증(일반기계·전기·화공 등), 기술사",
    "INFORMATION-TECHNOLOGY": "정보처리기사, AWS 인증 자격증, SQLD, ADsP",
    "BUSINESS-DEVELOPMENT": "경영지도사, PMP(프로젝트관리전문가)",
    "HR": "공인노무사, HRM전문가, PHR",
    "ACCOUNTANT": "전산세무회계, 재경관리사, CPA, AICPA",
    "FINANCE": "AFPK, CFP, CFA, 금융투자분석사",
    "HEALTHCARE": "간호사·의사 면허, 보건의료정보관리사",
    "SALES": "유통관리사, 물류관리사, ERP정보관리사",
    "TEACHER": "정교사 자격증, 한국어교원, 평생교육사",
    "DESIGNER": "시각디자인기사, 컴퓨터그래픽스운용기능사, 컬러리스트기사",
    "DIGITAL-MEDIA": "멀티미디어콘텐츠제작전문가, 사회조사분석사",
    "PUBLIC-RELATIONS": "브랜드관리사, 광고기획사",
    "CONSULTANT": "경영지도사, 가치평가사",
    "BANKING": "은행텔러, 외환전문역, 신용분석사",
    "CONSTRUCTION": "건축기사, 토목기사, 건설안전기사",
    "AGRICULTURE": "농화학기사, 종자기사, 유기농업기사",
    "AUTOMOBILE": "자동차정비기사, 자동차검사기사",
    "CHEF": "조리기능사(한식·양식·일식 등), 조리산업기사",
    "FITNESS": "생활스포츠지도사, 건강운동관리사",
    "AVIATION": "항공정비사, 항공무선통신사, 운항관리사",
    "ADVOCATE": "변호사, 법무사, 변리사",
    "APPAREL": "패션디자인산업기사, 의류기사, 패션머천다이징산업기사",
    "ARTS": "문화예술교육사, 예술치료사",
    "BPO": "CS Leaders(관리사), 테일러 자격증"
}

# 💡 [신규 추가] 직무별 주요 추천 기업 매핑 사전
job_companies = {
    "ENGINEERING": "삼성전자, 현대자동차, LG화학, SK하이닉스",
    "INFORMATION-TECHNOLOGY": "네이버, 카카오, 라인, 쿠팡, 배달의민족, 토스",
    "BUSINESS-DEVELOPMENT": "SK텔레콤, CJ ENM, 당근마켓, 무신사",
    "HR": "주요 대기업 인사팀, 글로벌 HR 컨설팅사, 리크루팅 전문 기업",
    "ACCOUNTANT": "삼일/삼정/안진/한영 회계법인, 대기업/중견기업 재무팀",
    "FINANCE": "미래에셋증권, 한국투자증권, 삼성자산운용",
    "HEALTHCARE": "서울대병원, 아산병원, 삼성서울병원, 셀트리온",
    "SALES": "이마트, 롯데쇼핑, GS리테일, 대기업 해외영업본부",
    "TEACHER": "메가스터디, 대교, 교원, 유명 사립학교 및 학원가",
    "DESIGNER": "이노션, 제일기획, 배달의민족 디자인실, IT 기업 UX팀",
    "DIGITAL-MEDIA": "CJ ENM, SBS, MBC, 유튜브 크리에이터 기획사(MCN)",
    "PUBLIC-RELATIONS": "프레인글로벌, 미디컴, 대기업 홍보실",
    "CONSULTANT": "맥킨지, BCG, 베인앤컴퍼니, 로컬 경영컨설팅사",
    "BANKING": "KB국민은행, 신한은행, 하나은행, 우리은행, 카카오뱅크",
    "CONSTRUCTION": "현대건설, 대우건설, GS건설, 삼성물산",
    "AGRICULTURE": "농협경제지주, 팜한농, CJ제일제당 바이오 부문",
    "AUTOMOBILE": "현대자동차, 기아, 현대모비스, HL만도",
    "CHEF": "CJ푸드빌, 현대그린푸드, 신세계푸드, 주요 특급호텔 F&B",
    "FITNESS": "국민체육진흥공단, 유명 피트니스 프랜차이즈, 스포츠 구단",
    "AVIATION": "대한항공, 아시아나항공, 제주항공, 인천국제공항공사",
    "ADVOCATE": "김앤장, 광장, 태평양, 세종 등 대형 로펌 및 기업 법무팀",
    "APPAREL": "삼성물산 패션부문, LF, 한섬, F&F, 무신사",
    "ARTS": "국립현대미술관, 예술의전당, CJ ENM 문화사업부",
    "BPO": "KTis, 효성ITX, 트랜스코스모스코리아"
}

# 한국어 기술 키워드 매핑 사전
keyword_mapping = {
    "파이썬": "Python", "데이터 분석": "Data Analysis", "데이터분석": "Data Analysis",
    "머신러닝": "Machine Learning", "머신 러닝": "Machine Learning",
    "딥러닝": "Deep Learning", "자바": "Java", "회계": "Accounting",
    "인사": "HR", "마케팅": "Marketing", "영업": "Sales", "디자인": "Design",
    "포토샵": "Photoshop", "일러스트": "Illustrator", "재무": "Finance",
    "개발": "Develop", "프로그래밍": "Programming", "엑셀": "Excel"
}

st.set_page_config(page_title="AI 직무 추천 시스템", page_icon="🤖")
st.title("🤖 AI 기반 직무 추천 시스템")
st.markdown("""
### 청년 구직자의 역량을 분석하여 적합한 직무를 추천합니다.
* **✔ AI 기반 직무 적합도 분석**
* **✔ 맞춤형 직무 추천 TOP 3 제공**
* **✔ 취업난 및 일자리 미스매치 문제 해결**
""")
st.write("---")
st.write("보유 역량과 경험을 입력하면 AI가 적합한 직무를 추천합니다.")

user_input = st.text_area("보유 역량 입력", placeholder="예: 파이썬, 데이터 분석, 머신러닝 (한글/영어 모두 가능)")

if st.button("직무 추천 받기"):
    if not user_input.strip():
        st.warning("보유 역량이나 경험을 먼저 입력해 주세요!")
    else:
        with st.spinner("역량을 분석하는 중입니다..."):
            processed_input = user_input
            for ko, en in keyword_mapping.items():
                processed_input = processed_input.replace(ko, en)
            
            try:
                translated_input = GoogleTranslator(source='auto', target='en').translate(processed_input)
            except:
                translated_input = processed_input
        
        text_vector = vectorizer.transform([translated_input])
        probs = model.predict_proba(text_vector)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        classes = model.classes_

        st.subheader("추천 직무 TOP 3")
        for rank, idx in enumerate(top3_idx, start=1):
            english_job = classes[idx]
            korean_job = job_korean.get(english_job, english_job)
            
            certs = job_certificates.get(english_job, "관련 자격증 정보 없음")
            # 💡 [신규 추가] 해당 직무의 추천 기업 정보 가져오기
            companies = job_companies.get(english_job, "관련 기업 정보 없음")
            
            # 결과 출력 (자격증과 추천 기업을 깔끔하게 정렬)
            if rank == 1:
              grade = "🌟 매우 적합"
            elif rank == 2:
              grade = "✅ 적합"
            else:
              grade = "📌 추천"

            st.write(f"{rank}위 : **{korean_job}** - {grade}")
            st.caption(f"> 🎖️ **추천 자격증:** {certs}")
            st.caption(f"> 🏢 **주요 추천 기업:** {companies}")
            st.write("")

