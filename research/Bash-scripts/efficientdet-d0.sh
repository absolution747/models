wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar -xvzf efficientdet_d0_coco17_tpu-32.tar.gz
sleep 5
mv efficientdet_d0_coco17_tpu-32 efficientdet
mv efficientdet/checkpoint efficientdet/checkpoint_start
