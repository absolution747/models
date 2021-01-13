import argparse

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import subprocess
import os


def parse_arguments():                                                                                                                                                                                                                                                
    #parser = argparse.ArgumentParser(description='')                                                                                                                                                                                                                  
    #parser.add_argument('pipeline')                                                                                                                                                                                                                                   
    #parser.add_argument('output')
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model", help = "Name of Eff-det model")
    ap.add_argument("-d","--data", help = "path to data")
    ap.add_argument("-b","--batch", help = "Batch size", type=int)
    ap.add_argument("-s","--steps", help = "Number of Epochs", type=int)                                                                                                                                                                                                                                    
    return ap.parse_args()                                                                                                                                                                                                                                        

def load_model(model):
    if model == 'D0':
        with open('Bash-scripts/efficientdet-d0.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D1':
        with open('Bash-scripts/efficientdet-d1.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D2':
        with open('Bash-scripts/efficientdet-d2.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D3':
        with open('Bash-scripts/efficientdet-d3.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D4':
        with open('Bash-scripts/efficientdet-d4.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D5':
        with open('Bash-scripts/efficientdet-d5.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D6':
        with open('Bash-scripts/efficientdet-d6.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)
    elif model == 'D7':
        with open('Bash-scripts/efficientdet-d7.sh', 'rb') as file:
            script = file.read()
        rc = subprocess.call(script, shell=True)



def main():                                                                                                                                                                                                                                                           
    args = parse_arguments()                                                                                                                                                                                                                                          
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    load_model(args.model)                                                                                                                                                                                                          
    print('model loaded')
    with tf.io.gfile.GFile('efficientdet/pipeline.config', "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)                                                                                                                                                                                                                 

    label_path = os.path.join(args.data, "label_map.pbtxt")
    tf_record_path = os.path.join(args.data, "default.tfrecord")

    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = tf_record_path 
    pipeline_config.train_input_reader.label_map_path = label_path
    pipeline_config.train_config.batch_size = args.batch
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.num_steps = args.steps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = args.steps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = 25
    pipeline_config.train_config.use_bfloat16 = False
    pipeline_config.train_config.fine_tune_checkpoint = "efficientdet/checkpoint_start/ckpt-0"

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile('efficientdet/pipline_new.config', "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text)                                                                                                                                                                                                                                          


if __name__ == '__main__':                                                                                                                                                                                                                                            
    main()
