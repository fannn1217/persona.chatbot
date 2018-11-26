import time
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
from models.dual_encoder import dual_encoder_model
import pandas as pd
from termcolor import colored
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

tf.flags.DEFINE_string("model_dir", "./runs/1542774662", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/persona/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

def get_features(context, persona, utterances):
  context_matrix = np.array(list(vp.transform([context])))
  persona_matrix = np.array(list(vp.transform([persona])))
  utterance_matrix = np.array(list(vp.transform([utterances[0]])))
  context_len = len(context.split(" "))
  persona_len = len(persona.split(" "))
  utterance_len = len(utterances[0].split(" "))
  features =  {
        "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
        "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
        "persona": tf.convert_to_tensor(persona_matrix, dtype=tf.int64),
        "persona_len": tf.constant(persona_len, shape=[1,1], dtype=tf.int64),
        "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
        "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
        "len":len(utterances)
  }

  for i in range(1,len(utterances)):
      utterance = utterances[i];

      utterance_matrix = np.array(list(vp.transform([utterance])))
      utterance_len = len(utterance.split(" "))

      features["utterance_{}".format(i)] = tf.convert_to_tensor(utterance_matrix, dtype=tf.int64)
      features["utterance_{}_len".format(i)] = tf.constant(utterance_len, shape=[1,1], dtype=tf.int64)

  return features, None

if __name__ == "__main__":
  # tf.logging.set_verbosity(tf.logging.INFO)
  # Load vocabulary
  vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
    FLAGS.vocab_processor_file)

  # Load data for predict
  test_df = pd.read_csv("./data/persona/predict.csv")
  #elementId = 0
  #INPUT_CONTEXT = test_df.Context[elementId]
  #INPUT_PERSONA = test_df.Persona[elementId]
  #POTENTIAL_RESPONSES = test_df.iloc[elementId,2:].values

  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  #starttime = time.time()
  for elementId in range(10):
    INPUT_CONTEXT = test_df.Context[elementId]
    INPUT_PERSONA = test_df.Persona[elementId]
    POTENTIAL_RESPONSES = test_df.iloc[elementId,2:].values
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, INPUT_PERSONA, POTENTIAL_RESPONSES))
    results = next(prob)
    print('\n')
    print(colored('[     Context]', on_color='on_blue',color="white"),INPUT_CONTEXT)
    print(colored('[     Persona]', on_color='on_blue',color="white"),INPUT_PERSONA)
    #print("[Results value ]",results)
    answerId = results.argmax(axis=0)
    if answerId==0:
        print(colored('[      Answer]', on_color='on_green'), POTENTIAL_RESPONSES[answerId])
    else:
        print (colored('[      Answer]', on_color='on_red'),POTENTIAL_RESPONSES[answerId])
        print (colored('[Right answer]', on_color='on_green'), POTENTIAL_RESPONSES[0])

  #endtime = time.time()
  print('\n')
  #print(colored('[Predict time]', on_color='on_blue',color="white"),"%.2f sec" % round(endtime - starttime,2))