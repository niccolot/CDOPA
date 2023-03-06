from matplotlib import pyplot as plt
import random

#number of depth images to keep, only a certain number contains the base gangli
zmin = 20
zmax = 30

#portion of image which contains the interested part of brain scan
xmin = 33
xmax = 162
ymin = 73
ymax = 202

deltaX = xmax-xmin
deltaY = ymax-ymin
deltaZ = zmax-zmin



def plot_slices(X_DATA, Y_DATA, patient_number=None):
    """
    plot 10 slices from a patient, the patient is chosen random
    if patient=None, else one has to put the number of the patient
    """

    n_imgs_rows = 2
    n_imgs_cols = 5
    
    fig, ax = plt.subplots(n_imgs_rows, n_imgs_cols)
    fig.subplots_adjust(wspace=0.1, hspace=0.05)

    #randomly chosen patient
    if patient_number == None:
        rand_index = random.randint(0, len(X_DATA)-1)

    for slice in range(deltaZ):

        plt.subplot(n_imgs_rows, n_imgs_cols, slice+1)
        
        if patient_number == None:
            plt.imshow(X_DATA[rand_index, :, :, slice])
            plt.axis('off')
            fig.suptitle('Patient: ' + str(rand_index)  + ', ' + 'label: ' + str(Y_DATA[rand_index]), fontsize=20)
            plt.title('Slice: ' + str(slice), fontsize=8)
        
        else:
            plt.imshow(X_DATA[patient_number, :, :, slice])
            plt.axis('off')
            fig.suptitle('Patient: ' + str(patient_number)  + ', ' + 'label: ' + str(Y_DATA[patient_number]), fontsize=20)
            plt.title('Slice: ' + str(slice), fontsize=8)
            
    plt.show()


def plot_loss_metric(loss, val_loss, metric, val_metric, metric_name='accuracy'):

    epochs_range = range(len(loss))

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    if metric_name == 'accuracy':
        plt.plot(epochs_range, metric, label='Training Accuracy')
        plt.plot(epochs_range, val_metric, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
    elif metric_name == 'recall':
        plt.plot(epochs_range, metric, label='Training Recall')
        plt.plot(epochs_range, val_metric, label='Validation Recall')
        plt.title('Training and Validation Recall')
    elif metric_name == 'precision':
        plt.plot(epochs_range, metric, label='Training Precision')
        plt.plot(epochs_range, val_metric, label='Validation Precision')
        plt.title('Training and Validation Precision')
    plt.legend(loc='lower right')
    

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()