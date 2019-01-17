import pandas as pd
from keras.engine.saving import model_from_json

import data

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(data.train_data_x, data.train_data_y, batch_size=32)
print(score)

predictions = [int(round(x[0])) for x in loaded_model.predict(data.test_data_x)]
id = data.full_test_data["PassengerId"]
submission = pd.DataFrame({
    "PassengerId": data.full_test_data["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)
