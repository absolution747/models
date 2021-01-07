import argparse

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def parse_arguments():                                                                                                                                                                                                                                                
    parser = argparse.ArgumentParser(description='')                                                                                                                                                                                                                  
    parser.add_argument('pipeline')                                                                                                                                                                                                                                   
    parser.add_argument('output')                                                                                                                                                                                                                                     
    return parser.parse_args()                                                                                                                                                                                                                                        


def main():                                                                                                                                                                                                                                                           
    args = parse_arguments()                                                                                                                                                                                                                                          
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()                                                                                                                                                                                                          

    with tf.io.gfile.GFile(args.pipeline, "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)                                                                                                                                                                                                                 

    pipeline_config.model.ssd.image_resizer.keep_aspect_ratio_resizer.min_dimension = 512                                                                                                                                                                                          
    pipeline_config.model.ssd.image_resizer.keep_aspect_ratio_resizer.max_dimension = 512                                                                                                                                                                                           
    
    pipeline_config.model.ssd.num_classes = 1
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = '/home/karan/Desktop/Annotation/default.tfrecord'
    pipeline_config.train_input_reader.label_map_path = '/home/karan/Desktop/Annotation/label_map.pbtxt'
    pipeline_config.train_config.batch_size = 1
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.use_bfloat16 = False
    pipeline_config.train_config.fine_tune_checkpoint = "efficientdet_d0_coco17_tpu-32/checkpoint_start/ckpt-0"

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(args.output, "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text)                                                                                                                                                                                                                                          


if __name__ == '__main__':                                                                                                                                                                                                                                            
    main()
