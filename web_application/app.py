import streamlit as st
import numpy as np
from detection.object_detection import detect_object, detect_dic
from detection.image_save import load_image
from siames.siames_network import preprocess_image, generate_embedding
import cv2
import os
from yolov7 import YOLOv7
from PIL import Image, ImageOps
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm
from skimage import io
from skimage.transform import resize

st.set_page_config(page_title="AI Style Recommender")

st.image('logo.png')

st.sidebar.title("AI Style Recommender")
st.sidebar.caption("Buy the style you want.")
st.sidebar.markdown("Made by Style Research Team")
st.sidebar.markdown("---")

# 여기서는 column의 형식으로 

st.sidebar.header("Our DB")
st.sidebar.image("https://media.giphy.com/media/l8DnhngGazFTPhNLK6/giphy.gif",width=300)
st.sidebar.markdown('---')


st.write('# Detect Your Style')

uploaded_image = st.file_uploader("이미지를 넣어주세요.", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    col1,col2 = st.columns(2)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    with col1:
      st.image(opencv_image,caption='입력한 이미지' ,channels="BGR")
    cv2.imwrite('image/query.jpg',opencv_image)
    image = detect_object(opencv_image)
    cv2.imwrite("image/Detected_objects.jpg",image)
    with col2:
      st.image('image/Detected_objects.jpg',caption='이미지 인식 결과')


    st.markdown('## Detection 결과')
    try: 
      detect = detect_dic(opencv_image)
      
      df1 = pd.DataFrame(detect) 
  
      # st.dataframe(df1, 800, 150)

      # 샴 모델 불러오기 
      
      if detect['class_id'] is not None:
              set_1 = set(detect['class_id'])
              set_2 = set(['001001','001005','001011'])
              set_3 = set(['002021','002007','002012'])
              set_4 = set(['003005','003002','003009'])
              set_5 = set(['005004','005011','018002'])
              if set_1.isdisjoint(set_2):
                  st.markdown('### 상의')
                  st.write('상의를 찾지 못하였습니다.')
                  st.markdown('---')
              else:
                for i in set_1.intersection(set_2):
                  st.markdown('### 상의')
                  Midclass = load_model("models/Midcateg_" + i + "_1216.h5", compile=False)
                  df = pd.read_csv("DB/df_styleReview_" + i + "_sim_matrix.csv", dtype=object)
              
                  # 관련된 미리 학습된 siamse df을 불러와 준다. 
                  
                  tqdm.pandas()
                  xmin = df1.loc[df1['class_id'] == i,'xmin'].values[0]
                  ymax = df1.loc[df1['class_id'] == i,'ymax'].values[0]
                  xmax = df1.loc[df1['class_id'] == i,'xmax'].values[0]
                  ymin = df1.loc[df1['class_id'] == i,'ymin'].values[0]
                  img = cv2.imread("image/query.jpg")
                  crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                  cv2.imwrite('image/croppedImage' + i + '.PNG',crop_img)
                  query_dir = 'image/croppedImage' + i + '.PNG' 
                  query = query_dir
                  # csv 파일을 통해 해주자. 
                  query_image = preprocess_image(query)
                  # 크롭된 이미지를 넣어주어서 불러온다. 
                  query_embedding_ = Midclass(query_image)[0].numpy().astype(np.float32).tolist()
                  query_embedding_ = str(query_embedding_)

                  df['distance_samImg_queryRev'] = df['embedding_sample'].progress_apply(lambda x: np.linalg.norm(np.asarray(eval(x), dtype=np.float32) - np.asarray(eval(query_embedding_), dtype=np.float32)))
                  df_sim_matrix = df.sort_values(by='distance_samImg_queryRev').reset_index(drop=True)
                  # 이밑은 image로 해준다. print은 write로 바꿔준다.
                  
                  image_ = io.imread(query)
                  image_resized = resize(image_, (200, 200))
                  st.image(image_resized,caption='Query Image') 
                      
                  col1,col2,col3,col4,col5 = st.columns(5)
                  with col1:
                    st.write('유사상품1')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[0, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  with col2:
                    st.write('유사상품2')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[1, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col3:
                    st.write('유사상품3')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[2, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col4:
                    st.write('유사상품4')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[3, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col5:
                    st.write('유사상품5')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[4, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True) 
                  st.markdown('---')

              if set_1.isdisjoint(set_3):
                  st.markdown('### 아우터')
                  st.write('아우터를 찾지 못하였습니다.')
                  st.markdown('---')
              else:
                for i in set_1.intersection(set_3):
                  st.markdown('### 아우터')
                  Midclass = load_model("models/Midcateg_" + i + "_1216.h5", compile=False)
                  df = pd.read_csv("DB/df_styleReview_" + i + "_sim_matrix.csv", dtype=object)
              
                  # 관련된 미리 학습된 siamse df을 불러와 준다. 
                  
                  tqdm.pandas()
                  xmin = df1.loc[df1['class_id'] == i,'xmin'].values[0]
                  ymax = df1.loc[df1['class_id'] == i,'ymax'].values[0]
                  xmax = df1.loc[df1['class_id'] == i,'xmax'].values[0]
                  ymin = df1.loc[df1['class_id'] == i,'ymin'].values[0]
                  img = cv2.imread("image/query.jpg")
                  crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                  cv2.imwrite('image/croppedImage' + i + '.PNG',crop_img)
                  query_dir = 'image/croppedImage' + i + '.PNG' 
                  query = query_dir
                  # csv 파일을 통해 해주자. 
                  query_image = preprocess_image(query)
                  # 크롭된 이미지를 넣어주어서 불러온다. 
                  query_embedding_ = Midclass(query_image)[0].numpy().astype(np.float32).tolist()
                  query_embedding_ = str(query_embedding_)

                  df['distance_samImg_queryRev'] = df['embedding_sample'].progress_apply(lambda x: np.linalg.norm(np.asarray(eval(x), dtype=np.float32) - np.asarray(eval(query_embedding_), dtype=np.float32)))
                  df_sim_matrix = df.sort_values(by='distance_samImg_queryRev').reset_index(drop=True)
                  # 이밑은 image로 해준다. print은 write로 바꿔준다.
                  
                  image_ = io.imread(query)
                  image_resized = resize(image_, (200, 200))
                  st.image(image_resized,caption='Query Image') 
                      
                  col1,col2,col3,col4,col5 = st.columns(5)
                  with col1:
                    st.write('유사상품1')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[0, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  with col2:
                    st.write('유사상품2')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[1, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col3:
                    st.write('유사상품3')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[2, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col4:
                    st.write('유사상품4')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[3, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col5:
                    st.write('유사상품5')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[4, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  st.markdown('---')  



              if set_1.isdisjoint(set_4):
                  st.markdown('### 하의')
                  st.write('하의를 찾지 못하였습니다.')
                  st.markdown('---')
              else:
                for i in set_1.intersection(set_4):
                  st.markdown('### 하의')
                  Midclass = load_model("models/Midcateg_" + i + "_1216.h5", compile=False)
                  df = pd.read_csv("DB/df_styleReview_" + i + "_sim_matrix.csv", dtype=object)
              
                  # 관련된 미리 학습된 siamse df을 불러와 준다. 
                  
                  tqdm.pandas()
                  xmin = df1.loc[df1['class_id'] == i,'xmin'].values[0]
                  ymax = df1.loc[df1['class_id'] == i,'ymax'].values[0]
                  xmax = df1.loc[df1['class_id'] == i,'xmax'].values[0]
                  ymin = df1.loc[df1['class_id'] == i,'ymin'].values[0]
                  img = cv2.imread("image/query.jpg")
                  crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                  cv2.imwrite('image/croppedImage' + i + '.PNG',crop_img)
                  query_dir = 'image/croppedImage' + i + '.PNG' 
                  query = query_dir
                  query_image = preprocess_image(query)
                  # 크롭된 이미지를 넣어주어서 불러온다. 
                  query_embedding_ = Midclass(query_image)[0].numpy().astype(np.float32).tolist()
                  query_embedding_ = str(query_embedding_)

                  df['distance_samImg_queryRev'] = df['embedding_sample'].progress_apply(lambda x: np.linalg.norm(np.asarray(eval(x), dtype=np.float32) - np.asarray(eval(query_embedding_), dtype=np.float32)))
                  df_sim_matrix = df.sort_values(by='distance_samImg_queryRev').reset_index(drop=True)
                  # 이밑은 image로 해준다. print은 write로 바꿔준다.
                  
                  image_ = io.imread(query)
                  image_resized = resize(image_, (200, 200))
                  st.image(image_resized,caption='Query Image') 
                  
                  
                  col1,col2,col3,col4,col5 = st.columns(5)
                  with col1:
                    st.write('유사상품1')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[0, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  with col2:
                    st.write('유사상품2')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[1, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col3:
                    st.write('유사상품3')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[2, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col4:
                    st.write('유사상품4')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[3, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col5:
                    st.write('유사상품5')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[4, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  st.markdown('---') 

          
              if set_1.isdisjoint(set_5):
                    st.markdown('### 신발')
                    st.write('신발을 찾지 못하였습니다.')
                    st.markdown('---')
              else:
                for i in set_1.intersection(set_5):
                  st.markdown('### 신발')
                  Midclass = load_model("models/Midcateg_" + i + "_1216.h5", compile=False)
                  df = pd.read_csv("DB/df_styleReview_" + i + "_sim_matrix.csv", dtype=object)
              
                  # 관련된 미리 학습된 siamse df을 불러와 준다. 
                  
                  tqdm.pandas()
                  xmin = df1.loc[df1['class_id'] == i,'xmin'].values[0]
                  ymax = df1.loc[df1['class_id'] == i,'ymax'].values[0]
                  xmax = df1.loc[df1['class_id'] == i,'xmax'].values[0]
                  ymin = df1.loc[df1['class_id'] == i,'ymin'].values[0]
                  img = cv2.imread("image/query.jpg")
                  crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                  cv2.imwrite('image/croppedImage' + i + '.PNG',crop_img)
                  query_dir = 'image/croppedImage' + i + '.PNG' 
                  query = query_dir
                  # 여기에 크롭된 이미지를 넣어주어야 한다. -> 추후 수정예정  
                  query_image = preprocess_image(query)
                  # 크롭된 이미지를 넣어주어서 불러온다. 
                  query_embedding_ = Midclass(query_image)[0].numpy().astype(np.float32).tolist()
                  query_embedding_ = str(query_embedding_)

                  df['distance_samImg_queryRev'] = df['embedding_sample'].progress_apply(lambda x: np.linalg.norm(np.asarray(eval(x), dtype=np.float32) - np.asarray(eval(query_embedding_), dtype=np.float32)))
                  df_sim_matrix = df.sort_values(by='distance_samImg_queryRev').reset_index(drop=True)
                  # 이밑은 image로 해준다. print은 write로 바꿔준다.
                  
                  image_ = io.imread(query)
                  image_resized = resize(image_, (200, 200))
                  st.image(image_resized,caption='Query Image') 
                  
                  col1,col2,col3,col4,col5 = st.columns(5)
                  with col1:
                    st.write('유사상품1')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[0, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  with col2:
                    st.write('유사상품2')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[1, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col3:
                    st.write('유사상품3')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[2, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col4:
                    st.write('유사상품4')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[3, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)   
                  with col5:
                    st.write('유사상품5')
                    row = df_sim_matrix.drop_duplicates("goods_id_pure",inplace=False).iloc[4, :]
                    st.image('image/' + load_image(row["goods_id_pure"]), caption = '상품이미지')
                    link = "https://musinsa.com/app/goods/" + row["goods_id_pure"]
                    product =  '[링크]' + '(' + link + ')'
                    st.markdown(product, unsafe_allow_html=True)
                  st.markdown('---')   
      else: 
            pass
    except:
      st.write('상품을 인식하지 못하였습니다.')
      st.markdown('---')       
else:
  st.write('이미지를 입력해 주세요.')
