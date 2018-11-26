from keras.models import model_from_json
import numpy as np
import os
import cv2


la = {"10":"A" , "11":"D" , "12":"T" , "13":"U" }
def Test(path):
    img = cv2.imread(path)
    input_image=img/255

    input_image = np.reshape(input_image, (1 , 36 , 36 , 3))


    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")


    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.predict_proba(input_image)


    if(np.amax(score)>0.70):
        if(np.argmax(score)>9):
           return (la[str(np.argmax(score))])
        else:
           return (np.argmax(score))



    else:

        return ("result not found")
