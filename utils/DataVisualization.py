import matplotlib.pyplot as plt
import math

class DataVisualization():

    def __init__(self, data_load):
        self.data_load = data_load

    def display_training_curves(self, training, validation, title, subplot):
        if subplot%10==1: 
            plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
            plt.tight_layout()
        ax = plt.subplot(subplot)
        ax.set_facecolor('#F8F8F8')
        ax.plot(training)
        ax.plot(validation)
        ax.set_title('model '+ title)
        ax.set_ylabel(title)
        #ax.set_ylim(0.28,1.05)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'valid.'])

    def batch_to_numpy_images_and_labels(self, data):
        images, labels = data
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        if numpy_labels.dtype == object: # binary string in this case,
                                        # these are image ID strings
            numpy_labels = [None for _ in enumerate(numpy_images)]
        # If no labels, only image IDs, return None for labels (this is
        # the case for test data)
        return numpy_images, numpy_labels

    def title_from_label_and_target(self, label, correct_label):
        if correct_label is None:
            return self.data_load.CLASSES[label], True
        correct = (label == correct_label)
        return "{} [{}{}{}]".format(self.data_load.CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                    self.data_load.CLASSES[correct_label] if not correct else ''), correct

    def display_one_flower(self, image, title, subplot, red=False, titlesize=16):
        plt.subplot(*subplot)
        plt.axis('off')
        plt.imshow(image)
        if len(title) > 0:
            plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
        return (subplot[0], subplot[1], subplot[2]+1)
        
    def display_batch_of_images(self, databatch, predictions=None):
        """This will work with:
        display_batch_of_images(images)
        display_batch_of_images(images, predictions)
        display_batch_of_images((images, labels))
        display_batch_of_images((images, labels), predictions)
        """
        # data
        images, labels = self.batch_to_numpy_images_and_labels(databatch)
        if labels is None:
            labels = [None for _ in enumerate(images)]
            
        # auto-squaring: this will drop data that does not fit into square
        # or square-ish rectangle
        rows = int(math.sqrt(len(images)))
        cols = len(images)//rows
            
        # size and spacing
        FIGSIZE = 13.0
        SPACING = 0.1
        subplot=(rows,cols,1)
        if rows < cols:
            plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
        else:
            plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
        
        # display
        for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
            title = '' if label is None else self.data_load.CLASSES[label]
            correct = True
            if predictions is not None:
                title, correct = self.title_from_label_and_target(predictions[i], label)
            dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
            subplot = self.display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
        
        #layout
        plt.tight_layout()
        if label is None and predictions is None:
            plt.subplots_adjust(wspace=0, hspace=0)
        else:
            plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
        plt.show()
        
    def display_confusion_matrix(self, cmat, score, precision, recall):
        plt.figure(figsize=(15,15))
        ax = plt.gca()
        ax.matshow(cmat, cmap='YlGn')
        ax.set_xticks(range(len(self.data_load.CLASSES)))
        ax.set_xticklabels(self.data_load.CLASSES, fontdict={'fontsize': 7})
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        ax.set_yticks(range(len(self.data_load.CLASSES)))
        ax.set_yticklabels(self.data_load.CLASSES, fontdict={'fontsize': 7})
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        titlestring = ""
        if score is not None:
            titlestring += 'f1 = {:.3f} '.format(score)
        if precision is not None:
            titlestring += '\nprecision = {:.3f} '.format(precision)
        if recall is not None:
            titlestring += '\nrecall = {:.3f} '.format(recall)
        if len(titlestring) > 0:
            ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
        plt.show()