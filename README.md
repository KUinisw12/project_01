# 12조 STYLE Research Team **AI STYLE RECOMMENDATION** 
----
## 프로젝트 진행상황 Log 

### 공태영
  - **완료**
    - Object Detection 모델(개인 colab에서 실행)
    - DB를 받은 후 csv 형식으로 변경하기 
    - Object Detection 학습 
      - Object Detection(Yolo v7/ Yolo v5) 진행 및 Fine Tuning - Modanet dataset, 자체 어노테이션 Test data set 활용
    - WEB 제작(그래프 활용/디자인 개선 진행중/ 구동 및 디자인)
      - Yolov7 학습 weight을 가져와 onnx파일로 변환 뒤 streamlit을 통해 웹사이트에서 object detection 모델 구현 시도 
        - 현재 sample 학습 weight로는 구현 성공 
    - WEB 제작(그래프 활용/디자인 개선 진행중/ 구동 및 디자인)
      - Siamese network 결과를 가져와서 상의, 아우터, 하의, 신발 카테고리별(col1,col2,col3,.. 형식) 유사한 상품 이미지(해당 상품 이미지에 link을 embedding)를 제안하는 웹사이트 구현

**MODELS** 받을 수 있는 주소 
   [drive link](https://drive.google.com/file/d/19yuG4zq2mTh5iABGEXiQ5ep4EfnTr_jI/view?usp=sharing)
   

### 박준영
- 날짜 : 12/08
  - DB 구축 
    -{상의 : 001, 아우터 : 002, 바지 : 003, 신발 : 005, 모자 : 007, 스니커즈 : 018, 원피스 : 020, 스커트 : 022} 
- 날짜 : 12/09
  - 크롤링, DB 완료, EDA 미완료
- 날짜 : 12/10
  - 수집DB EDA, 그래프 만드는 중.
  - Object Detection 샘플이미지에 대해서 잘 안 되는 것 확인. 전략 수정 필요.
  - Siamese network 학습 중(OD 필요없는 샘플이미지와 포토리뷰이미지에 대해.)
- 참고자료
  - Similarity https://89douner.tistory.com/334
  - https://simonezz.tistory.com/43
  - tensorboard Embedding Projector https://towardsdatascience.com/taking-the-tensorboard-embedding-projector-to-the-next-level-bde53deb6bb7
  - https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905
- To Do
  - 포토리뷰, 스타일리뷰 EDA(상품별 리뷰이미지 개수, 이미지 특성 등)
  - Positive set 구성, Negative set 구성(코드 만들었음)
  - DB 수정(카테고리) OD결과 집어 넣을 수 있도록 DB 확장
  - 발표자료에 들어갈 도표, 이미지비준비

prepare data for siames
image/anchor/ positive / negative csv
process 

1. object detection(anchor extraction-EACH DB Category)
2. make DB goods code/sample url/ review url/ anchor(xmin ymin xmax ymax)
3. make triplet train set(positive input/ negative input/ anchor input)
4. embeddings-generation
5. visualizing for presentation
6. make website
- 웹에 들어갈 그래프 EDA 준비

### 이석우

  - Object Detection 모델 2
  - db object detection(detectron2- faster/mask rcnn/yolo5 / fine tuning) & feature extraction
  - data for siames
image/anchor/ positive / negative csv
process 
1. object detection(anchor extraction-EACH DB Category)
2. make DB goods code/sample url/ review url/ anchor(xmin ymin xmax ymax)
3. make triplet train set(positive input/ negative input/ anchor input)
4. embeddings-generation
5. visualizing for presentation
6. make website
- 옷 입히기 프로세스 추가(상품 사진을 자신의 체형 이미지에 입혀보기)/ - 다양한 모델 실험 필요 / 향후 진행 예정
- WEB 제작(wix사용 UI 구축- 그래프 활용/ 디자인 개선 진행중/ 구동 및 디자인/연결 및 테스트)
- GIT 정리

### 정명민
- PPT 제작 및 전체 진행 총괄
- 날짜 : 12/08
  - Siamese Network pipeline 실행(데스크탑 PC에서 실행)
  - DB 실행
  - To Do
  - 무신사 DB를 이용한 object detection data와 Siamese Network 연결 
  - 코드 정리 및 PPT 정리 진행(ppt & design)
  - 발표 계획 및 전략 수립
- 날짜 : 12/09
  - Siamese Network 전체적인 flow 따라서 파이썬 파일 진행
  - 다른 파트(Object Detection, Siamese Network, MobileNetv2 통한 DB 작성하는 부분) flow up
----
## 발표 계획
- 파일 업로드, 유사 샘플 찾아주는 웹페이지 구현(향후 추천된 옷을 자신의 체형과 비슷한 이미지에 입혀보는 프로세스까지 구현 예정) 
- similarity Embedding Vector Space (tensorboard, pandas 등 활용)
- Hitmap
- 데이터 그래프
- 기타 멋져 보이는 것들 조사(논문 참고)

---
## 진행 계획
- 12.11까지 AI STYLE RECOMMENDATION 모델 구현 완료
- 12.12~12.17까지 UI구성
- 12.12~12.17까지 PPT구성
- 12.19 발표자료 제출
- 12.21 총 발표
---
## OD 범위 
상의 
반소매(001001)/맨투맨(001005) /민소매(001011)

아우터
베스트 (002021)/ 겨울 싱글 코트(002007)
/숏패딩(002012)

바지
레깅스(003005) 데님 팬츠(003002) 숏팬츠(003009)

신발
샌들(005004) 부츠(005011) 캔버스단화(018002)

---
## 코드
각 브랜치에(개인) 흩어져 있는 파일들을 마스터 브랜치에 정리해서 올려놓을 예정(프로젝트 진행 순)
