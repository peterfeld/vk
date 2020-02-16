import vk_api
import requests
import pandas as pd
import vk
from urllib.request import urlretrieve
import os
from datetime import date,timedelta
import json
import datetime

session = requests.Session()
login, password = '79256581763', '3myS2DX'
vk_session = vk_api.VkApi(login, password)
try:
    vk_session.auth(token_only=True)
except vk_api.AuthError as error_msg:
    print(error_msg)

app_id, login, password = '6145959', '79256581763', '3myS2DX'
session = vk.AuthSession(app_id, login, password)
vk = vk.API(session, v='5.62')

# API-ключ созданный ранее
token = "c900bb4d9d7657a238e8580eeb91aba9f22df0a811b435bcccfb806bd214c166b845fbf7b9f8f787b17df"
# Авторизуемся как сообщество
vk_group = vk_api.VkApi(token=token)


import math
import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
from tqdm import tqdm
from time import time

import requests
from bs4 import BeautifulSoup

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

global avg_score
global write_msg

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces

def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def main_photo_id(user_id):
    try:
        print(user_id)
    
        response = vk.users.get(user_ids=user_id,
                    fields = 'photo_id')
        photo_id = response[0]['photo_id']
    
        return photo_id
    except:
        return (None)

def load_embeddings_1_user_2_photos(user_id):
    """Возвращает эмбеддинг главного фото со страницы по идентификатору пользователя"""

    try:
        photo_id = main_photo_id(user_id)
        photo = vk.photos.getById(photos=photo_id)

        photo_user_id = photo[0]['owner_id']
        photo_604 = photo[0]['photo_604']

    except:
        photo_user_id = 'no_photo_user_id'
        photo_604 = 'no_photo_url'

    len_1 = 0 # Кол-во лиц
    embeddings_1 = [] # Эмбеддинги

    try:
        photo_path = str(int(photo_user_id))
        urlretrieve(photo_604,photo_path) # Выгрузка главного фото

        img = cv2.imread(photo_path) # Считали фото
        face = getFace(img) # Сделали эмбеддинг

        len_1 = len(face)
        if len_1>0:
            embeddings_1 = face[0]['embedding']

        os.remove(photo_path) # Удаляем фото

    except:
        try:
            # Предположим, что это закрытый профиль, пробуем выгрузить по прямой ссылке
            photo_path = str(user_id)

            url = 'https://www.vk.com/'+str(user_id) # url для второй страницы
            
            html_text = vk_session.http.get(url)
            soup = BeautifulSoup(html_text.text)
            main_photo_url = soup.find('img', {'class': 'page_avatar_img'})['src']

            urlretrieve(main_photo_url,photo_path) # Выгрузка главного фото

            img = cv2.imread(photo_path) # Считали фото
            face = getFace(img) # Сделали эмбеддинг

            len_1 = len(face)
            if len_1>0:
                embeddings_1 = face[0]['embedding']

            os.remove(photo_path) # Удаляем фото
        except:
            len_1 = len_1

    len_2 = 0 # Кол-во лиц
    embeddings_2 = [] # Эмбеддинги

    try:
        # Пытаемся выгрузить еще одно фото пользователя
        photos = vk.photos.get(owner_id=int(photo_user_id), album_id='profile') # Все фото профиля
        url= photos['items'][-2]['photo_1280'] # Url главного фото
        photo_path = str(int(photo_user_id)) + '_2'
        urlretrieve(url,photo_path) # Выгрузка главного фото

        img = cv2.imread(photo_path) # Считали фото
        face = getFace(img) # Сделали эмбеддинг

        len_2 = len(face)
        if len_2>0:
            embeddings_2 = face[0]['embedding']

        os.remove(photo_path) # Удаляем фото

    except:
        len_2 = len_2

    emb_df = pd.DataFrame({'id':[photo_user_id],'len_1':[len_1],'embeddings_1':[embeddings_1],
                                                'len_2':[len_2],'embeddings_2':[embeddings_2]})
    
    return emb_df

def find_closest_by_2_photos(user_id,target_df,df_path='Data/Precalculated_Embeddings/final_photos_for_pilot_15_04_19.pkl'):
    """Возвращает ссылки на страницы с наиболее похожими лицами на лицо у выбранного пользователя"""
    
    df = pd.read_pickle(df_path)
    np_all = np.array(df.embeddings)
    np_all = np.concatenate(np_all, axis=0 )

    # To Do проверка того, что найдено лицо хотя бы на одном фото как в этой функции (find_closest_by_2_photos),
    # Так и в главной функции, тем самым расширим
    # Развилка: либо сравнивать 2 к 1 и искать лучшее, либо 2 к 2 и искать, наверное, надо протестить
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    if target_df.len_1[0]==1:
        target_emb = target_df.embeddings_1[0]
        diff = np_all-target_emb
        diff_in_2 = diff**2
        quadr_sum = diff_in_2.sum(axis=1)
        df['distance'] = np.sqrt(quadr_sum)
        df1 = df.copy()

    if target_df.len_2[0]==1:
        target_emb = target_df.embeddings_2[0]
        diff = np_all-target_emb
        diff_in_2 = diff**2
        quadr_sum = diff_in_2.sum(axis=1)
        df['distance'] = np.sqrt(quadr_sum)
        df2 = df.copy()

    df = pd.concat([df1,df2])

    df = df[['id','distance']].groupby('id',as_index=False).mean()
    df = df.sort_values('distance')
    result = df.head(2000)
    result['id'] = result['id'].astype(int).astype(str)

    return result

def recommend_1_photo(cl_photo_path,user_id,target_id,user_text,keyboard_to_photo):

    recommend_photo_done = False
    while recommend_photo_done == False:
        try:
            cl_photo_df = pd.read_pickle(cl_photo_path)
            girl_id = cl_photo_df.id.values[0]
            cl_photo_df[1:].to_pickle(cl_photo_path) # удалим первую, уже выбранную, строчку
            photo_id = main_photo_id(girl_id)

            message_to_photo = 'Вот страничка девушки https://vk.com/id' + str(girl_id)

            vk_group.method('messages.send', {'user_id': user_id, 
                                        'message': message_to_photo,
                                        'attachment': 'photo'+photo_id,
                                        'keyboard': json.dumps(keyboard_to_photo,ensure_ascii=False)})


            to_history = pd.DataFrame({'id':[user_id],'time':[int(time())],
               'state':['3_2'],'message_from_u':[user_text],
               'meassage_to_u':[message_to_photo],'target_id':[target_id]}) 

            recommend_photo_done = True
        except:
            pass

    return to_history

def write_msg(user_id, message, keyboard=None):
    
    if keyboard==None:
        vk_group.method('messages.send', {'user_id': user_id, 
                                    'message': message})
    else:
        vk_group.method('messages.send', {'user_id': user_id, 
                                    'message': message,
                                    'keyboard':json.dumps(keyboard,ensure_ascii=False)})
def msg_recomendation (user_id,keyboard,recomendation_profile_predict,avg_score):
    global write_msg
    target_id=recomendation_profile_predict(user_id,write_msg,avg_score)
    target_foto_id=main_photo_id(target_id)
    if target_id is not None:
        if target_foto_id is not None:
            vk_group.method('messages.send', {'user_id': user_id, 
                       'message': 'Вот страничка девушки https://vk.com/id' + str(target_id),
                        'attachment': 'photo'+target_foto_id,
                        'keyboard': json.dumps(keyboard,ensure_ascii=False)}) 
        else:
            vk_group.method('messages.send', {'user_id': user_id, 
                       'message': 'Вот страничка девушки https://vk.com/id' + str(target_id),
                        'keyboard': json.dumps(keyboard,ensure_ascii=False)}) 
    else:
        vk_group.method('messages.send', {'user_id': user_id, 
           'message': 'К сожалению нам не удалось сделать рекомендацию',
            'keyboard': json.dumps(keyboard_back_error,ensure_ascii=False)}) 
    return (target_id)

def get_main_id_num(x):
    return (vk.users.get(user_ids=x)[0]['id'])




