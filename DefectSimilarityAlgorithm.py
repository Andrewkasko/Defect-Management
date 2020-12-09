# Andrew Kaskaniotis 07/05/2019
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


global DefectCount
global NewDefectID
global predictedSeverity #Severity
global severitySum


#SelectedDefect = input("Enter Defect Description: ")
SelectedDefect = "Migrated Vehicle records does not have Vehicle License State field populated."

severitySum = 0
predictedSeverity = 0.0
SimilarityThreshold = 0.3


os.environ['TFHUB_CACHE_DIR'] = 'insert tfcache dir'


#embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/1")
embed = hub.Module("https://tfhub.dev/google/elmo/2")
read = pd.read_excel('insert excel file dir') # change this directory for desired dataset

#Adjust variables to appropriate columns for dataset

DefectIDs = read.iloc[:, 1].tolist()
Severity = read.iloc[:, 2].tolist()
AssignedTo = read.iloc[:, 4].tolist()
messages = read.iloc[:, 6].tolist()

tf.logging.set_verbosity(tf.logging.ERROR)

def CreateNewDefect(str):
    messages.append(str)
    global SelectedDefectIndex
    SelectedDefectIndex = int(len(messages)) - 1

def FindSelectedDefect(str):
    global SelectedDefectIndex
    if SelectedDefect in messages:
        SelectedDefectIndex = messages.index(str)
    else:
        CreateNewDefect(SelectedDefect)

FindSelectedDefect(SelectedDefect)

def FindDefectID(str):
    DefectRow = messages.index(str)
    DefectIDSingle = read.iloc[DefectRow,1]
    return DefectIDSingle

def sortFirst(val):
    return val[0]

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  SimilarityScores = np.asarray(corr[SelectedDefectIndex,:])
  Desciptions = np.asarray(messages)

  SM = list(zip(SimilarityScores, DefectIDs, Severity, Desciptions ))
  SMSorted = list(zip(SimilarityScores, DefectIDs, Severity, Desciptions))
  SMSorted.sort(key = sortFirst,reverse = True)

  #Top 5 Defects
  print("Defect: " + SelectedDefect)
  print("Top 5 similar defects: ")
  global severitySum
#  severitySum = 0
 # predictedSeverity = 0.0
  for z in range(1, 6):

    if SMSorted[z][0] > SimilarityThreshold:
        print(str(z) +". " + str(SMSorted[z]))
        severitySum = severitySum + SMSorted[z][2] #Severity
        predictedSeverity = round(severitySum / z)

        if z == 5:
            print("Predicted Severity: " + str(predictedSeverity))
    else:
        if z > 1:
            print("Predicted Severity: " + str(predictedSeverity))
            return
        else:
            print("No defects found above the similarity threshold " + str(SimilarityThreshold))
            return

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(messages))
  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))

def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages_})
  plot_similarity(messages_, message_embeddings_, 90)

# Similarity Visualised
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  run_and_plot(session, similarity_input_placeholder, messages,
               similarity_message_encodings)

sts_input1 = tf.placeholder(tf.string, shape=(None))
sts_input2 = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)
cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
