# A-APOS v3.0

## 실행 방법

### 1. SMT 데이터 넣기
```
AAPOS/
└── SMT_2020 - Final/
    └── AutoSched/
        ├── dataset 1/
        ├── dataset 2/
        ├── dataset 3/
        └── dataset 4/
```

### 2. 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Streamlit Cloud 배포
1. GitHub에 push (SMT 데이터 제외)
2. share.streamlit.io → New app → 레포 연결
3. Main file: `app.py` → Deploy
