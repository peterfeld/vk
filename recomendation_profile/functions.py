from settings import login_name, password_name,features,features_func
from settings import features,columns_for_df,top_groups,top_type_groups
import vk_api
import pandas as pd
import numpy as np
import time


##### functions_for_1user

def main(spis):
    """ Функция для получения информации по пользователям вк с определенным фильтром """

    login, password = login_name, password_name
    vk_session = vk_api.VkApi(login, password)

    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)
        return

    vk = vk_session.get_api()
    response_id = vk.users.get(user_id=spis,
                            fields = 'about, activities, bdate, *\
                            blacklisted, books, can_post, can_see_all_posts, *\
                            can_see_audio, can_send_friend_request, can_write_private_message, *\
                            career, city, connections, contacts, counters, country, domain, *\
                            education, first_name_{nom}, followers_count, games, *\
                            has_mobile, has_photo, home_town, interests, last_name_{nom}, last_seen, *\
                            maiden_name, military, movies, music, nickname, occupation, *\
                            online, personal, photo_id, quotes, relatives, relation, schools, *\
                            screen_name, sex, site, status, timezone, tv, universities, verified, *\
                            wall_comments')
    response_friends = vk.friends.get(user_id=spis,
                            fields = 'sex, bdate, city')
    response_groups = vk.groups.get(user_id=spis,fields=['activity'],
                            extended=True)
    return response_id,response_friends,response_groups

def ggjson( b, delim ):
    val = {}
    for i in b.keys():
        if isinstance( b[i], dict ):
            get = ggjson( b[i], delim )
            for j in get.keys():
                val[ i + delim + j ] = get[j]
        elif isinstance( b[i], list) and len(b[i])!=0:
            if isinstance( b[i][0],dict):
                get = ggjson( b[i][0], delim )
                for j in get.keys():
                    val[ i + delim + j ] = get[j]    
            else:
                val[i] = b[i]
        else:
            val[i] = b[i]
    return val

def json_transform(response_id):
    json_massive=[]
    for ids in range(len(response_id)):   
        json_massive.append(response_id[ids])
    g=[]
    for ind in range(len(json_massive)):
        g.append([])
        g[ind]=ggjson(json_massive[ind], "__" )
    for k in g[0].keys():
        if type(g[0][k])==list: g[0][k]=0  
    return (g)

def json_transform_2(response):
    json_massive=[]
    for ids in range(len(response['items'])):   
        json_massive.append(response['items'][ids])
    s=[]
    for ind in range(len(json_massive)):
        s.append([])
        s[ind]=ggjson(json_massive[ind], "__" )
    return (s)

def number_phone(x):
    if x==x and x!='':
        if ((x[0]=='8') & (len(x)==11))|((x[0:2]=='+7') & (len(x)==12)):
            return 'Нормальный'
        else: return 'Не нормальный'
    else: return np.nan
    
def df_for_1User(response_id):
    df=pd.DataFrame(json_transform(response_id))
    df_for_user=pd.DataFrame(index=[0])
    ### день, месяц, год рождения
    if 'bdate' in df.columns:
        bd=pd.to_datetime(df['bdate'],format="%d.%m.%Y")[0]
        df_for_user.loc[:,'year'] = bd.year
        df_for_user.loc[:,'month'] = bd.month
        df_for_user.loc[:,'day'] = bd.day
    bool_col=list(features[(features.Take=='yes')&(features.Type=='bool')]['All_columns'])
    int_col=list(features[(features.Take=='yes')&(features.Type=='int')]['All_columns'])
    df_for_user=pd.concat([df_for_user,df[[i for i in bool_col+int_col if i in list(df.columns)]]],axis=1)
    ### ОБРАБАТЫВАЕМ Str
    str_col=list(set(features[(features.Take=='yes')&(features.Type=='str')]['All_columns'])-set(features_func))
    df_str=df[[i for i in str_col if i in list(df.columns)]]
    df_str=pd.DataFrame([[1 if df_str.loc[:,i].values!='' else 0 for i in list(df_str.columns) ]],columns=df_str.columns)
    ### ОБРАБАТЫВАЕМ func
    if 'domain' in df.columns: df_str.loc[:,'domain']=np.where(df.domain.str.startswith('id')!=-1,1,0)
    if 'mobile_phone' in df.columns:df.loc[:,'mobile_phone']=df.mobile_phone.map(number_phone)    
    ### ОБРАБАТЫВАЕМ cat  
    cat_col=list(features[(features.Take=='yes')&(features.Type=='cat')]['All_columns'])+['mobile_phone']
    df_cat=df[[i for i in cat_col if i in list(df.columns)]].copy()
    for i in df_cat.columns:
        try:  df_cat.loc[:,i]=df_cat[i].astype(float).astype(str)
        except:  df_cat.loc[:,i]=df_cat[i].astype(str)
    df_cat=pd.get_dummies(df_cat)
    df_for_user=pd.concat([df_for_user,df_cat,df_str],axis=1) 
    return (df_for_user)
        
    
def friends(response_friends):
    try:
        friends_id=pd.DataFrame(json_transform_2(response_friends))
        friends={'0':friends_id.loc[friends_id.sex==0,'id'].size,
                '1':friends_id.loc[friends_id.sex==1,'id'].size,
                '2':friends_id.loc[friends_id.sex==2,'id'].size}
        friends=pd.DataFrame(list(friends.values()),index=friends.keys()).T
        friends.columns=['1.0','2.0','0.0']
        return (friends)
    except:
        return (pd.DataFrame([[np.nan,np.nan,np.nan]],columns=['1.0','2.0','0.0']))

def groups (response_groups):
    groups_id=pd.DataFrame(json_transform_2(response_groups))
    groups_df=pd.DataFrame(top_groups)
    groups_df.loc[groups_df[0].isin(groups_id.screen_name.values),'kol']=1
    groups_df.fillna(0,inplace=True)
    groups_df=groups_df.set_index(0)
    groups_df.columns=[0]
    groups_df=groups_df.T
    # типы групп
    type_groups_id=groups_id.groupby('activity')['id'].count().reset_index()
    type_groups_model_df=pd.DataFrame(top_type_groups)
    type_groups_id.columns=[0,'id']
    type_groups_model_df=type_groups_model_df.merge(type_groups_id,how='left',on=0)
    type_groups_model_df.fillna(0,inplace=True)
    type_groups_model_df=type_groups_model_df.set_index(0)
    type_groups_model_df.columns=[0]
    type_groups_model_df=type_groups_model_df.T
    return (groups_df,type_groups_model_df)

def likes(likes_total_1user,id_search):
    likes_tot_cols_1user = ['likes_gr_' + str(i) for i in range(1,len(top_groups)+1)]
    try:
        likes_total_df=pd.DataFrame(data=likes_total_1user.loc[[id_search]].values,columns=likes_tot_cols_1user)
        likes_total_df.fillna(0,inplace=True)
    except:
        likes_total_df=pd.DataFrame(columns=likes_tot_cols_1user)
    return (likes_total_df)

def model_df_for_user_wo_w2v(id_search,likes_total_1user):
    try:
        if isinstance(id_search,int):
            response_id,response_friends,response_groups=main(id_search)
            df=df_for_1User(response_id)
            groups_df,type_groups_model_df=groups(response_groups)
            friends_df=friends(response_friends)
            likes_total_df=likes(likes_total_1user,id_search)

            df_final=pd.DataFrame(columns_for_df).set_index(0).join(df.T,how='left')

            df_user=pd.concat([df_final.T,friends_df,likes_total_df,groups_df,type_groups_model_df],axis=1)
            ###
            sex=['_m'] if df_user.sex[0]==2 else ['_w']
            df_user.columns = [y+x for x,y in zip(sex*len(df_user.columns), df_user.columns)]
            df_user['key'] = 1
            df_user=df_user.replace('',np.nan)
            df_user.fillna(0,inplace=True)
            return (df_user)
    except:
            return (np.nan)

##### functions_df_for_model

def friends_sex_func(friends_sex_f):
    friends_sex = pd.read_csv(friends_sex_f,dtype={'id':'int64','female_friends':'int16','male_friends':'int16','pct_of_female_friends':'float16'})
    friends_sex.columns=['id']+list(friends_sex.columns[1:])
    return (friends_sex)

def woman_pro_in_relation_filt_f(woman_prob_in_relation_filt_f):
    global woman_pro_in_relation_filt
    woman_prob_in_relation_filt = pd.read_csv(woman_prob_in_relation_filt_f)
    woman_prob_in_relation_filt = woman_prob_in_relation_filt[woman_prob_in_relation_filt.probably_in_relation==0]
    woman_prob_in_relation_filt = woman_prob_in_relation_filt.loc[:,['id']]
    return woman_prob_in_relation_filt

def filters_name_friends_f(good_names_filt_f,friends_filt_f):
    good_names_filt = pd.read_csv(good_names_filt_f) # Для фильтра по имени, выбасываем слишком редкие имена
    friends_filt = pd.read_csv(friends_filt_f) # Для фильтрации по друзьям
    friends_filt = friends_filt.loc[:,['id']]
    return (good_names_filt,friends_filt)

def groups_type_groups_f(topNGroups_f,topNType_f):
    svod_groups_total = pd.read_csv(topNGroups_f)
    svod_groups_total.rename(columns={'-':'groups_other'}, inplace=True)
    svod_groups_total.columns=['id']+list(svod_groups_total.columns[1:])
    svod_type_groups_total = pd.read_csv(topNType_f)
    svod_type_groups_total.rename(columns={'-':'group_types_other'}, inplace=True)
    svod_type_groups_total.columns=['id']+list(svod_type_groups_total.columns[1:])
    svod_groups_total = svod_groups_total.fillna(0)
    svod_type_groups_total = svod_type_groups_total.fillna(0)
    return (svod_groups_total,svod_type_groups_total)

def likes_df(likes_f):
    likes_total=pd.read_csv(likes_f,header=None)
    likes_tot_cols = ['likes_gr_' + str(i) for i in range(0,len(likes_total.columns))]
    likes_tot_cols[0] = 'id'
    likes_total.columns = likes_tot_cols
    likes_total = likes_total.drop_duplicates('id')
    likes_total.fillna(0,inplace=True)
    return likes_total

def add_likes_groups(d,friends_sex,likes_total,svod_groups_total,svod_type_groups_total):
    d = d.merge(friends_sex,on='id',how='left')
    d = d.merge(likes_total,on='id',how='left')
    d = d.merge(svod_groups_total,on='id',how='left')
    d = d.merge(svod_type_groups_total,on='id',how='left')  
    return d

def df_for_predict(df,required_files):
    """
    Делаем собираем в один датафрейм выгрузку по пользователям, группам, лайкам и типам групп
    """
    df_model_f,model_f,features_for_model_f,likes_f,\
good_names_filt_f,friends_filt_f,woman_prob_in_relation_filt_f,friends_sex_f,topNGroups_f,topNType_f=required_files

    good_names_filt,friends_filt=filters_name_friends_f(good_names_filt_f,friends_filt_f)

    friends_sex=friends_sex_func(friends_sex_f)
    woman_pro_in_relation_filt=woman_pro_in_relation_filt_f(woman_prob_in_relation_filt_f)
    likes_total=likes_df(likes_f)
    svod_groups_total,svod_type_groups_total=groups_type_groups_f(topNGroups_f,topNType_f)

    # проверка на фильтры имен и друзей
    df = pd.merge(df,good_names_filt,on='id')
    df = pd.merge(df,friends_filt,on='id')
    
    df = add_likes_groups(df,friends_sex,likes_total,svod_groups_total,svod_type_groups_total)
    
    df=df.replace('',np.nan)
    df.fillna(0,inplace=True)
    return df

###### function_for_predict


def women_not_in_relation_f(df,woman_prob_in_relation_filt_f):
    # Женщины не в отношениях
    woman_prob_in_relation_filt = woman_pro_in_relation_filt_f(woman_prob_in_relation_filt_f)
    # выбираем женщин
    women_not_in_relation = df[((df.relation==1) | (df.relation==6) | (df.relation==0) | (df.relation.isnull())) & (df.sex==1)]
    women_not_in_relation = pd.merge(women_not_in_relation,woman_prob_in_relation_filt,on='id')

    cols = women_not_in_relation.columns
    len_cols = len(cols)
    women_not_in_relation.columns = [y+x for x,y in zip(['_w']*len_cols, cols)]
    women_not_in_relation['key'] = 1
    return (women_not_in_relation)

def predict_1_user(id_search,data_for_predict,files_for_1_user,model_info):
    start=time.monotonic()
    loaded_model,list_features_correct=model_info
    unique_men_df=model_df_for_user_wo_w2v(id_search,files_for_1_user)
    print('Выгрузка пользователя закончена... : {:>9.2f}'.format(time.monotonic()-start))
    if type(unique_men_df)==pd.core.frame.DataFrame:
        try:
            row=data_for_predict.shape[0]
            col=unique_men_df.shape[1]   
            X  = pd.DataFrame(np.repeat(unique_men_df.values,row,axis=1).reshape(col,row)).T
            X=X.join(data_for_predict)
            X.columns=list(unique_men_df)+list(data_for_predict.columns)
            print('Выбор девушек... : {:>9.2f}'.format(time.monotonic()-start))
            X.drop('key',axis=1,inplace=True)
            id_w = X.id_w.values
            X=X[list_features_correct]
            print('Предсказание : {:>9.2f}'.format(time.monotonic()-start))
            y_pred = loaded_model.predict_proba(X)[:,1]
            print('Сбор результата... : {:>9.2f}'.format(time.monotonic()-start))
            user_predict = pd.DataFrame({'id':id_w,'score':y_pred})   
            user_predict.id=user_predict.id.astype(int)
            user_predict['client_id'] = id_search 
            #
            mean_score=user_predict.score.mean()
            #
            user_predict = user_predict.sort_values('score',ascending=False).head(100)
            print('Расчет окончен... : {:>9.2f}'.format(time.monotonic()-start))
            return (user_predict,mean_score)
        except:
            print ('ошибка предикт')
            return(np.nan,np.nan)
        
def normalise_score(x,mean_score):
    if x<mean_score: return (x/mean_score*0.5)
    else: return (1/(1+np.exp(-x/mean_score+1)))