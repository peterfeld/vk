import pandas as pd
import numpy as np
import os.path
import pickle
#import time
from sklearn.externals import joblib
from settings import df_model_f,model_f,features_for_model_f,likes_f,\
good_names_filt_f,friends_filt_f,woman_prob_in_relation_filt_f,friends_sex_f,topNGroups_f,topNType_f,group_type_group_f,\
features,columns_for_df,top_groups,top_type_groups,df_for_predict_f,path_to_save

from functions import df_for_predict,predict_1_user,women_not_in_relation_f,model_df_for_user_wo_w2v,main,df_for_1User,groups,friends,likes
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

likes_total_1user = pd.read_csv(likes_f,header=None,index_col=0)

data_for_predict = pd.read_pickle(df_for_predict_f)#.head(10000)

loaded_model = joblib.load(model_f)
list_features_correct=list(pd.read_csv(features_for_model_f)['0'].values)
model_info=[loaded_model,list_features_correct]

def recomendation_profile_predict(id_name):
    path_user=path_to_save+'\\'+str(id_name)+'.pkl'
    if os.path.exists(path_user):
        user_recomendations=pd.read_pickle(path_user)
    else:
        rec_profile=predict_1_user(id_name,data_for_predict,likes_total_1user,model_info).id[:100]
        #rec=['https://vk.com/id'+str(x) for x in rec_profile.values]
        user_recomendations=pd.DataFrame({'rec_profile':rec_profile,'saw':0})
    
    results=user_recomendations.loc[user_recomendations.saw==0,'rec_profile'].values[0]
    user_recomendations.loc[user_recomendations.rec_profile==results,'saw']=1
    user_recomendations.to_pickle(path_user)
    if results is not None: return (results)
    else:
        user_recomendations.loc[:,'saw']=0
        user_recomendations.to_pickle(path_user)
        return(None)
    

    