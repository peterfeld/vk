import os.path
import warnings

import pandas as pd
from functions import predict_1_user,model_df_for_user_wo_w2v
from settings import model_f, features_for_model_f, likes_f, \
    df_for_predict_f, path_to_save
# import time
from sklearn.externals import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)

likes_total_1user = pd.read_csv(likes_f, header=None, index_col=0)

data_for_predict = pd.read_pickle(df_for_predict_f)  # .head(10000)

loaded_model = joblib.load(model_f)
list_features_correct = list(pd.read_csv(features_for_model_f)['0'].values)
model_info = [loaded_model, list_features_correct]
###
path_score_2_users = path_to_save + '\\score_2_users.pkl'

global avg_score,user_score_2_users

def recomendation_profile_predict(id_name):
    global avg_score
    path_user = path_to_save + '\\' + str(id_name) + '.pkl'
    if os.path.exists(path_user):
        user_recomendations = pd.read_pickle(path_user)      
    else:
        write_msg(user_id,'Подожди пожалуйста, собираем информацию...')
        rec_profile,mean_score = predict_1_user(id_name, data_for_predict, likes_total_1user, model_info)
        rec_profile=rec_profile.id[:100]
        # rec=['https://vk.com/id'+str(x) for x in rec_profile.values]
        user_recomendations = pd.DataFrame({'rec_profile': rec_profile, 'saw': 0})
        #
        avg_score=avg_score.append(pd.DataFrame({'id_user':[id_name], 'avg_score': [mean_score]}))
        avg_score.to_pickle(path_avg_score)
    if len(user_recomendations.loc[user_recomendations.saw == 0, 'rec_profile'].values)>0:
        results = user_recomendations.loc[user_recomendations.saw == 0, 'rec_profile'].values[0]
        user_recomendations.loc[user_recomendations.rec_profile == results, 'saw'] = 1
        user_recomendations.to_pickle(path_user)
    else:
        user_recomendations.loc[:, 'saw'] = 0
        results = user_recomendations.loc[user_recomendations.saw == 0, 'rec_profile'].values[0]
        user_recomendations.to_pickle(path_user)
    if results is not None:
        return (results)
    
def recomendation_score_2_users(id_name,id_target):
    global avg_score,user_score_2_users
    try:
        path_score_2_users = path_to_save + '\\score_2_users.pkl'
        path_avg_score=path_to_save + '\\avg_score.pkl'
        path_user = path_to_save + '\\' + str(id_name) + '.pkl'
        if os.path.exists(path_user):
            mean_score=avg_score.loc[avg_score.id_user==id_name,'avg_score'][0]
        else:
            rec_profile,mean_score=predict_1_user(id_name, data_for_predict, likes_total_1user, model_info)
            rec_profile=rec_profile.id[:100]
            user_recomendations = pd.DataFrame({'rec_profile': rec_profile, 'saw': 0})
            #
            avg_score=avg_score.append(pd.DataFrame({'id_user':[id_name], 'avg_score': [mean_score]}))
            avg_score.to_pickle(path_avg_score)
            user_recomendations.to_pickle(path_user)
        woman_df=model_df_for_user_wo_w2v(id_target,likes_total_1user)
        res,_ = predict_1_user(id_name, woman_df, likes_total_1user, model_info)
        score=normalise_score(res.score[0],mean_score)
        
        score_2_users = pd.DataFrame({'id_user': [id_name], 'id_target':[id_target], 'score': [score]})
        
        user_score_2_users=user_score_2_users.append(score_2_users)
        user_score_2_users.to_pickle(path_score_2_users)
        return (score)
    except:
        return (None)