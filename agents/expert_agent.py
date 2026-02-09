class ExpertAgent:
    def critical_cases(self, predictions, outliers):
        critical = []

        for i in range(len(predictions)):
            if predictions[i] == 1 and outliers[i]:
                critical.append(i)

        return critical
