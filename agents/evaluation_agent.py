from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class EvaluationAgent:
    def show_confusion_matrix(self, model, X_test, y_test, title):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(cm).plot()
        plt.title(title)
        plt.show()
