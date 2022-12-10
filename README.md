# 12SW **Style Danawa** 
----
## 프로젝트 진행상황 Log 

### 공태영
- 날짜 : 12/08
  - Object Detection 모델(개인 colab에서 실행)
  - DB를 받은 후 csv 형식으로 변경하기

### 박준영
- 날짜 : 12/08
  - DB 구축 
    -{상의 : 001, 아우터 : 002, 바지 : 003, 신발 : 005, 모자 : 007, 스니커즈 : 018, 원피스 : 020, 스커트 : 022} 
- 날짜 : 12/09
  - 크롤링, DB 완료, EDA 미완료
- 참고자료
  - Similarity https://89douner.tistory.com/334
- To Do
  - 포토리뷰, 스타일리뷰 EDA(상품별 리뷰이미지 개수, 이미지 특성 등)
  - Positive set 구성, Negative set 구성
- 날짜: 12/10

prepare data for siames
image/anchor/ positive / negative csv
      (bbox) (same goods code) (another goods code)
process 

1. object detection(anchor extraction-EACH DB Category)
2. make DB goods code/sample url/ review url/ anchor(xmin ymin xmax ymax)
3. make triplet train set(positive input/ negative input/ anchor input)
4. embeddings-generation
5. visualizing for presentation
6. make website

### 이석우
- 날짜 : 12/08 
  - Object Detection 모델 2
- 날짜 : 12/09
  - db object detection & feature extraction
- 날짜 : 12/10
  - data for siames
image/anchor/ positive / negative csv
      (bbox) (same goods code) (another goods code)
process 

1. object detection(anchor extraction-EACH DB Category)
2. make DB goods code/sample url/ review url/ anchor(xmin ymin xmax ymax)
3. make triplet train set(positive input/ negative input/ anchor input)
4. embeddings-generation
5. visualizing for presentation
6. make website


### 정명민
- 날짜 : 12/08
  - Siamese Network pipeline 실행(데스크탑 PC에서 실행)
    - "https://github.com/chirag4798/Shop-The-Look" 참조
  - DB 실행
  - To Do
    - 무신사 DB를 이용한 object detection data와 Siamese Network 연결 
    
----
## 발표 계획
- 파일 업로드, 유사 샘플 찾아주는 웹페이지 구현
- similarity Embedding Vector Space (tensorboard, pandas 등 활용)
- Hitmap
- 데이터 그래프
- 기타 멋져 보이는 것들 조사(논문 참고)

---
## 진행 계획
- 12.11까지 스타일 다나와 모델 구현 완료
- 12.12~12.16까지 UI구성
- 12.12~12.16까지 PPT구성

