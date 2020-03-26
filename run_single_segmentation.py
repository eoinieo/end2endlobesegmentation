import sys
import os
import time
import futils.util as futil
import segmentor as v_seg
import tensorflow as tf
# suppress a lot of deprecation warnings coming from tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras.backend as K

K.set_learning_phase(1)

import logging
logger = logging.getLogger(__name__)
if len(logger.handlers) == 0:
    LEVEL = logging.INFO
    logger.setLevel(LEVEL)
    formatter = logging.Formatter('[%(asctime)-15s] [%(levelname)8s] %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#LOAD THE MODEL
segment = v_seg.v_segmentor(batch_size=1, model='models/final.h5', ptch_sz=128, z_sz=64)

#GET THE IMAGE PATH
try:
    path_to_input_scan_file = sys.argv[1]
except IndexError as err:
    logger.error('Supply the path to the image file as the only CLI arg, e.g.')
    logger.error('>> pipenv run python run_single_segmentation.py path/to/image.nii.gz')
    path_to_input_scan_file = 'data/006_ref_raw_f64.nii.gz'
    path_to_input_scan_file = 'data/003_GST_CTce_raw_crop.nii.gz'
finally:
    assert os.path.isfile(path_to_input_scan_file), "ERROR no such input file as {}".format(path_to_input_scan_file)

#LOAD THE CT_SCAN
ct_scan, origin, shape, spacing, orientation = futil.load_itk(path_to_input_scan_file, get_orientation=True)
if (orientation[-1] == -1):
    ct_scan = ct_scan[::-1]
logger.info('origin: {} {} {}'.format(*origin))
logger.info('shape: {} {} {}'.format(*shape))
logger.info('spacing: {} {} {}'.format(*spacing))
logger.info('orientation: {} {} {} {} {} {} {} {} {}'.format(*orientation))

#NORMALIZATION
ct_scan = futil.normalize(ct_scan)

#PREDICT the segmentation
t1 = time.time()
lobe_mask = segment.predict(ct_scan)
t2 = time.time()
logger.info('prediction runtime (s): %i', int(t2-t1))

#Save the segmentation
path_to_output_scan_file = path_to_input_scan_file.replace('.nii', '_sumLobes.nii')
futil.save_itk(path_to_output_scan_file, lobe_mask, origin, spacing)
