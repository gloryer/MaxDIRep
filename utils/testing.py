
from sklearn.metrics import accuracy_score




def test(generator, classifier, x_source_test, x_target_test, y_source_test, y_target_test):
    y_source_pred_test = classifier.predict(generator(x_source_test)).argmax(1)
    y_target_pred_test = classifier.predict(generator(x_target_test)).argmax(1)

    # prediction accuracy
    accuracy_source = accuracy_score(y_source_test.argmax(1), y_source_pred_test)
    accuracy_target = accuracy_score(y_target_test.argmax(1), y_target_pred_test)

    track = ('Testing ==> \n'
             'Acc Source: {} Acc Target: {}\n').format(accuracy_source, accuracy_target)

    print(track)



