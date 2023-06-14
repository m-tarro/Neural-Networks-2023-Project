import re
import numpy as np
import tensorflow as tf

# Data pipelines and exploration functions based on notebook: https://www.kaggle.com/code/achinih/flower-classification-cnn-models

class DataLoad(tf.data.TFRecordDataset):
    
    def __init__(self, image_size=512, batch_size=16):
        
        self.AUTO = tf.data.experimental.AUTOTUNE
        
        self.BATCH_SIZE = batch_size
        
        assert image_size in (192,224,331,512)
        self.IMAGE_SIZE = [image_size, image_size]
        
        self.GCS_PATH = f'/kaggle/input/tpu-getting-started/tfrecords-jpeg-{image_size}x{image_size}'
        self.TRAINING_FILENAMES = tf.io.gfile.glob(self.GCS_PATH + '/train/*.tfrec')
        self.VALIDATION_FILENAMES = tf.io.gfile.glob(self.GCS_PATH + '/val/*.tfrec')
        self.TEST_FILENAMES = tf.io.gfile.glob(self.GCS_PATH + '/test/*.tfrec')
        
        self.NUM_TRAINING_IMAGES = self.count_data_items(self.TRAINING_FILENAMES)
        self.NUM_VALIDATION_IMAGES = self.count_data_items(self.VALIDATION_FILENAMES)
        self.NUM_TEST_IMAGES = self.count_data_items(self.TEST_FILENAMES)
        
        self.TRAINING_STEPS_PER_EPOCH = self.NUM_TRAINING_IMAGES // self.BATCH_SIZE
        self.TEST_STEPS_PER_EPOCH = self.NUM_TEST_IMAGES // self.BATCH_SIZE
        
        self.CLASSES = ['pink primrose',        'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',      'wild geranium',         # 00 - 04
                        'tiger lily',           'moon orchid',               'bird of paradise', 'monkshood',      'globe thistle',         # 05 - 09
                        'snapdragon',           "colt's foot",               'king protea',      'spear thistle',  'yellow iris',           # 10 - 14
                        'globe-flower',         'purple coneflower',         'peruvian lily',    'balloon flower', 'giant white arum lily', # 15 - 19
                        'fire lily',            'pincushion flower',         'fritillary',       'red ginger',     'grape hyacinth',        # 20 - 24
                        'corn poppy',           'prince of wales feathers',  'stemless gentian', 'artichoke',      'sweet william',         # 25 - 29
                        'carnation',            'garden phlox',              'love in the mist', 'cosmos',         'alpine sea holly',      # 30 - 34
                        'ruby-lipped cattleya', 'cape flower',               'great masterwort', 'siam tulip',     'lenten rose',           # 35 - 39
                        'barberton daisy',      'daffodil',                  'sword lily',       'poinsettia',     'bolero deep blue',      # 40 - 44
                        'wallflower',           'marigold',                  'buttercup',        'daisy',          'common dandelion',      # 45 - 49
                        'petunia',              'wild pansy',                'primula',          'sunflower',      'lilac hibiscus',        # 50 - 54
                        'bishop of llandaff',   'gaura',                     'geranium',         'orange dahlia',  'pink-yellow dahlia',    # 55 - 59
                        'cautleya spicata',     'japanese anemone',          'black-eyed susan', 'silverbush',     'californian poppy',     # 60 - 64
                        'osteospermum',         'spring crocus',             'iris',             'windflower',     'tree poppy',            # 65 - 69
                        'gazania',              'azalea',                    'water lily',       'rose',           'thorn apple',           # 70 - 74
                        'morning glory',        'passion flower',            'lotus',            'toad lily',      'anthurium',             # 75 - 79
                        'frangipani',           'clematis',                  'hibiscus',         'columbine',      'desert-rose',           # 80 - 84
                        'tree mallow',          'magnolia',                  'cyclamen ',        'watercress',     'canna lily',            # 85 - 89
                        'hippeastrum ',         'bee balm',                  'pink quill',       'foxglove',       'bougainvillea',         # 90 - 94
                        'camellia',             'mallow',                    'mexican petunia',  'bromelia',       'blanket flower',        # 95 - 99
                        'trumpet creeper',      'blackberry lily',           'common tulip',     'wild rose']                               # 100 - 103
    
    def count_data_items(self, filenames):
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

    #processing the images into floats from 0,1 and reshaping to the size required for a TPU.
    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [*self.IMAGE_SIZE, 3])
        return image
    
    #reading the labels for my images and returns a dataset with the image and label in a pair.
    def read_labeled_tfrecord(self, example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        label = tf.cast(example['class'], tf.int32)
        return image, label
    
    #reading the unlabeled data to use for testing.
    def read_unlabeled_tfrecord(self, example):
        UNLABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        idnum = example['id']
        return image, idnum

    #Reading multiple files at once to improve performance. 
    #Ordering data order decreases the speed and as the data will be shuffled later on anyways. 
    def load_dataset(self, filenames, labeled=True, ordered=False):
    
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disabling order

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTO)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self.read_labeled_tfrecord if labeled else self.read_unlabeled_tfrecord, num_parallel_calls=self.AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset
    
    def onehot_classes(self, image, label):
        class_no = len(self.CLASSES)
        return image, tf.one_hot(label, class_no)
    
    def get_training_dataset(self, image_augment=False, batch_augment=False, ordered=False, onehot=True, split='train', **kwargs):
        dataset = self.load_dataset(self.TRAINING_FILENAMES, labeled=True, ordered=ordered)
        dataset = dataset.repeat(10)
        if split == 'train:
            dataset = dataset.take(int(self.NUM_TRAINING_IMAGES * 0.8))
        elif split == 'test':
            dataset = dataset.skip(int(self.NUM_TRAINING_IMAGES * 0.8))
        if image_augment:
            dataset = dataset.map(image_augment, num_parallel_calls=self.AUTO)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
        if batch_augment:
            dataset = dataset.batch(self.BATCH_SIZE)
            dataset = dataset.map(lambda x, y: batch_augment([x, y], onehot=onehot), num_parallel_calls=self.AUTO)
            dataset = dataset.unbatch()
        elif onehot:
            dataset = dataset.map(self.onehot_classes, num_parallel_calls=self.AUTO)
        if not ordered:
            dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO) # get next batch while training
        return dataset
    
    def get_validation_dataset(self, ordered=False, onehot=False):
        dataset = self.load_dataset(self.VALIDATION_FILENAMES, labeled=True, ordered=ordered)
        dataset = dataset.batch(self.BATCH_SIZE)
        if onehot: 
            dataset = dataset.map(self.onehot_classes, num_parallel_calls=self.AUTO)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO)
        return dataset
    
    def get_test_dataset(self, ordered=True):
        dataset = self.load_dataset(self.TEST_FILENAMES, labeled=False, ordered=ordered)
        # dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO)
        return dataset
