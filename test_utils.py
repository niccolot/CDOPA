from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

def multiclass_roc_auc_score(y_test, y_pred, target_list, multiclass=True, average="macro"):
    #returns roc_auc score and plots roc curve
    plt.figure(figsize=(6, 6))

    if multiclass:
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)

        for (idx, c_label) in enumerate(target_list):
            fpr, tpr, thresholds = roc_curve(y_test[idx, :], y_pred[idx, :])
            plt.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))

    else:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label='AUC:%0.2f' % auc(fpr, tpr))

    plt.plot(fpr, fpr, 'b-', label='Random Guessing')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return roc_auc_score(y_test, y_pred, average=average)



def get_errors(x_test, test_probs, pred_probs, num_errors=4):
    """
    x_test is test dataset

    test_probs and pred_probs are respectively the one hot encoded test labes and 
    the predicted probabilities vectors (arranged in a tensor) that the sofmtax gives at test time

    num_errors is how many misslabeled images to show
    """

    test_labels = np.argmax(test_probs, axis=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    error_labels = (pred_labels - test_labels != 0)

    pred_probs_errors = pred_probs[error_labels]
    test_labels_errors = test_labels[error_labels]
    pred_labels_errors = pred_labels[error_labels]
    x_test_errors = x_test[error_labels]

    # Probabilities of the wrong predicted numbers
    y_pred_errors_prob = np.max(pred_probs_errors, axis=1)

    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(pred_probs_errors, test_labels_errors, axis=1))

    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

    # Sorted list of the delta prob errors
    sorted_delta_errors = np.argsort(delta_pred_true_errors)

    # Top errors
    most_important_errors = sorted_delta_errors[-num_errors:]

    return most_important_errors, pred_probs_errors, test_labels_errors, x_test_errors, pred_labels_errors


def display_errors(target_dict, x_test, test_labels, predictions, nrows=2, ncols=2):
    """
    target_dict is a dictionary that associates numbers from 0 to num classes to
    the name of the classes e.g.

    target_dict = {
        '0': class_0,
        '1': class_1,
     ...etc
    }
    """

    n = 0
    fig, ax = plt.subplots(nrows, ncols)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    errors_index, \
    pred_probs_errors, \
    test_labels_errors, \
    img_errors, \
    pred_labels_errors = get_errors(x_test, test_labels, predictions)

    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow(img_errors[error])
            ax[row, col].set_title("Predicted label: {}\nTrue label: {}".format(target_dict[str(pred_labels_errors[error])], 
                                                                                target_dict[str(test_labels_errors[error])]))
            n += 1

    plt.show()
