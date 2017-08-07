#!/usr/bin/env python2.7

'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container


 
python speechEmo_client.py --server=localhost:9000 --sound=demo.wav


'''

from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

import scipy.io.wavfile
import numpy as np 
import librosa
import librosa.display

import threading 


# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import classification_pb2



# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000','PredictionService host:port')
tf.app.flags.DEFINE_string('sound', '', 'path to sound in wav format')


# These need to match the constants used when training the emo model
tf.app.flags.DEFINE_string('n_input', 26, 'Number of MFCC features')
tf.app.flags.DEFINE_string('n_context', 9, 'Number of frames of context')
FLAGS = tf.app.flags.FLAGS



def do_inference(hostport, audio2):
     
    print (audio2)
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    keys = np.asarray([1, 2, 3])
    keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)

    
    features_tensor_proto = tf.contrib.util.make_tensor_proto(audio2,dtype=tf.float32)

    print ("Sending Classify request...")
    # Prepare request
    #request = classification_pb2.ClassificationRequest()
    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'emo'
    request.model_spec.signature_name = 'predict_sound'

    
    # tf.contrib.util.make_tensor_proto(values, dtype= NOne, shape=None)
    request.inputs['Sounds'].CopyFrom(keys_tensor_proto)

           
    #result = stub.Classify(request, 180.0)  # 10 seconds
    result = stub.Predict(request, 10.0)  # 10 seconds
   
    print(result)

def main(_):

    if not FLAGS.server:
        print ('please specify server host:port')
        return
    if not FLAGS.sound:
        print ('please specify an audio file')
        return

    features = np.empty((0,180))

    # See prediction_service.proto for gRPC request/response details.
    X, sample_rate = librosa.load(FLAGS.sound)
    #print(FLAGS.sound)
    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
   
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        
    sound = np.hstack([mfccs,chroma,mel])
    features = np.vstack([features,sound])
    
    audio = np.float32(features)
    
    do_inference(FLAGS.server, audio)
 

    '''
    # Generate inference data
    keys = numpy.asarray([1, 2, 3])
    keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)
    features = numpy.asarray(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    features_tensor_proto = tf.contrib.util.make_tensor_proto(features,dtype=tf.float32)

    # Create gRPC client and request
   
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'emo'
    request.model_spec.version.value = 1
    request.inputs['keys'].CopyFrom(keys_tensor_proto)
    
     
    # Send request
    result = stub.Predict(request, 120.0)
    print(result)
    
    
    # Send request
    with open(FLAGS.sound, 'rb') as f:
        #print (f)

        # See prediction_service.proto for gRPC request/response details.
        X, sample_rate = librosa.load(FLAGS.sound)
        print(FLAGS.sound)
        stft = np.abs(librosa.stft(X))

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        
        Sounds = np.hstack([mfccs,chroma,mel])
                
        request = predict_pb2.PredictRequest()
        #request = classification_pb2.ClassificationRequest()

        # Call emotion model to make classfication of the speech 
        request.model_spec.name = 'emo'
        request.model_spec.signature_name = 'classfy_emo'
        request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(Sounds))

        
        #result = stub.Classify(request, 60.0)  # 60 secs timeout
        result = stub.Predict(request, 60.0)  # 5 seconds
        #print(result)
    '''
if __name__ == '__main__':
    tf.app.run()
