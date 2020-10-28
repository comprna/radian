import numpy as np
from tensorflow.keras.models import load_model

def main():
  # Overall algorithm:
    # Get test data
    # Load finalized model
    # Pass test data into network
    # CTC decoding of network outputs
    # Assemble into reads
    # Compute error rate

  saved_filepath = './saved_model'
  model = load_model(saved_filepath, compile = True)

  samples_to_predict = np.array([])

  predictions = model.predict(samples_to_predict)

  # CTC decoding

  # Assembly

  # Error rate



  # Refs:
    # https://www.programcreek.com/python/example/122027/keras.backend.ctc_decode
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    # https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/

  print("Test")

if __name__ == "__main__":
  main()