message_all_func = """Привет! Выбери вид рекомендации девушек:
          """

keyboard_all_func = { 
    "one_time": True, 
    "buttons": [ 
    [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"1\"}", 
          "label": "Для отношений" 
        }, 
        "color": "positive"
      }, 
     { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"2\"}", 
          "label": "По фото" 
        }, 
        "color": "positive" 
      }], 
     [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"8\"}", 
          "label": "Выход" 
        }, 
        "color": "negative" 
      }]
    ] 

  } 


keyboard_to_photo = { 
    "one_time": True, 
    "buttons": [ 
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"1\"}", 
          "label": "1. Что написать?" 
        }, 
        "color": "positive"
      }, 
     { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"2\"}", 
          "label": "2. Следующую девушку" 
        }, 
        "color": "positive" 
      }],
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"3\"}", 
          "label": "3. Похожие на другую" 
        }, 
        "color": "positive"
      }, 
     { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"4\"}", 
          "label": "4. Главное меню" 
        }, 
        "color": "negative" 
      }
      
      ]
    ] 
  } 

keyboard_what_to_write_next_5_or_back = { 
    "one_time": True, 
    "buttons": [ 
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"1\"}", 
          "label": "Что написать?" 
        }, 
        "color": "positive"
      }, { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"2\"}", 
          "label": "Следующие 5" 
        }, 
        "color": "positive"
      }], 
     [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"3\"}", 
          "label": "Назад" 
        }, 
        "color": "negative" 
      }]
    ] 
  } 

keyboard_back = { 
    "one_time": True, 
    "buttons": [ 
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"3\"}", 
          "label": "Назад" 
        }, 
        "color": "negative" 
      }]
    ] 
  } 


keyboard_next_5_or_back = { 
    "one_time": True, 
    "buttons": [ 
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"2\"}", 
          "label": "Следующие 5"
        }, 
        "color": "positive"
      }, 
     { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"3\"}", 
          "label": "Назад" 
        }, 
        "color": "negative" 
      }]
    ] 
  } 


keyboard_what_to_write = { 
    "one_time": True, 
    "buttons": [ 
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"1\"}", 
          "label": "1. Следующую девушку" 
        }, 
        "color": "positive" 
      }],
      [{ 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"2\"}", 
          "label": "2. Похожие на другую" 
        }, 
        "color": "positive"
      }, 
     { 
        "action": { 
          "type": "text", 
          "payload": "{\"button\": \"3\"}", 
          "label": "3. Главное меню" 
        }, 
        "color": "negative" 
      }
      ]
    ] 

  } 

message_2_1 = """
Вот девушки, которые наилучше подходят тебе для отношений по мнению нашего алгоритма:
https://vk.com/helenahuddy
https://vk.com/nastydudnik
https://vk.com/id136135636
https://vk.com/id18206470
https://vk.com/id2121969
"""

# Варианты состояний пользователя:
# 0 - либо новый пользователь, либо вышедший из бота
# 1 - стартовое меню: от 1 до 8
# 2_1 - пользователь выбрал 1 в стартовом меню, отправили ему 5 страниц девушек. (keyboard: еще 5 девушек, либо назад)
# 3_1 - пользователь выбрал 2 или 3 в стартовом меню, попросили пользователя отправить фото или ссылку на страницу ВК
# 3_2 - отправили пользователю фото похожей на эталон девушки с текстом и кнопками

#message_ask_photo = """Пришли в ответном сообщении фото девушки, либо ссылку на её профиль вконтакте."""
message_ask_photo = """Пришли в ответном сообщении ссылку на профиль девушки вконтакте."""

message_what_to_write = """Можешь написать ей:
                    - Привет) Ты не поверишь, искусственный интеллект выбрал тебя для знакомства со мной!
                    - Привет! Ты очень похожа на одну мою знакомую, мы могли видеться раньше?
                    """
message_end = """Пока) возвращайся!"""