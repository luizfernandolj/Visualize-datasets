import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from quantifiers.ClassifyCountCorrect.MAX import MAX
from quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from quantifiers.ClassifyCountCorrect.T50 import T50
from quantifiers.ClassifyCountCorrect.X import Xqtf
from quantifiers.DistributionMatching.DyS import DyS
from quantifiers.DistributionMatching.HDy import HDy
from quantifiers.DistributionMatching.SORD import SORD

class Experiments:
    def __init__(self, train_test, name, niterations, batch_sizes, alphas, clf, thr):
        self.train_test = train_test
        self.name = name
        self.niterations = niterations
        self.batch_sizes = batch_sizes
        self.alphas = alphas
        self.clf = clf
        self.thr = thr
        # self.quantifiers = ["CC", "ACC", "PCC", "PACC", "X", "MAX", "T50", "MS", "HDy", "DyS", "SORD"]
        self.quantifiers = ["CC", "ACC", "PCC", "PACC", "X", "MAX", "T50", "MS", "HDy", "DyS", "SORD"]
        self.quantifiers_initialized = {}
        self.measure = ['topsoe', 'probsymm', 'hellinger']
        columns = ["name", "sample", "Test_size", "alpha", "actual_prop", "pred_prop", "abs_error", "quantifier", "threshold"]
        self.table = pd.DataFrame(columns=columns)

    def apply_quantifier(self, quantifier, clf, thr, measure, train, test):
        if quantifier not in self.quantifiers_initialized:
            if quantifier == "CC":
                cc = ClassifyCount(classifier=clf, threshold=thr)
                cc.fit(train[0], train[1])
                self.quantifiers_initialized["CC"] = cc

                return cc.predict(test)
            if quantifier == "ACC":
                acc = AdjustedClassifyCount(classifier=clf, threshold=thr)
                acc.fit(train[0], train[1])
                self.quantifiers_initialized["ACC"] = acc

                return acc.predict(test)
            if quantifier == "PCC":
                pcc = ProbabilisticClassifyCount(classifier=clf)
                pcc.fit(train[0], train[1])
                self.quantifiers_initialized["PCC"] = pcc

                return pcc.predict(test)

            if quantifier == "PACC":
                pacc = ProbabilisticAdjustedClassifyCount(classifier=clf, threshold=thr)
                pacc.fit(train[0], train[1])
                self.quantifiers_initialized["PACC"] = pacc

                return pacc.predict(test)

            if quantifier == "X":
                x_qtf = Xqtf(classifier=clf)
                x_qtf.fit(train[0], train[1])
                self.quantifiers_initialized["X"] = x_qtf

                return x_qtf.predict(test)

            if quantifier == "MAX":
                max_qtf = MAX(classifier=clf)
                max_qtf.fit(train[0], train[1])
                self.quantifiers_initialized["MAX"] = max_qtf

                return max_qtf.predict(test)

            if quantifier == "T50":
                t50 = T50(classifier=clf)
                t50.fit(train[0], train[1])
                self.quantifiers_initialized["T50"] = t50

                return t50.predict(test)

            if quantifier == "MS":
                ms = MedianSweep(classifier=clf)
                ms.fit(train[0], train[1])
                self.quantifiers_initialized["MS"] = ms

                return ms.predict(test)

            if quantifier == "HDy":
                hdy = HDy(classifier=clf)
                hdy.fit(train[0], train[1])
                self.quantifiers_initialized["HDy"] = hdy

                return hdy.predict(test)

            if quantifier == "DyS":
                dys = DyS(classifier=clf, similarity_measure=measure)
                dys.fit(train[0], train[1])
                self.quantifiers_initialized["DyS"] = dys

                return dys.predict(test)

            if quantifier == "SORD":
                sord = SORD(classifier=clf)
                sord.fit(train[0], train[1])
                self.quantifiers_initialized["SORD"] = sord

                return sord.predict(test)
        else:
            return self.quantifiers_initialized[quantifier].predict(test)

    def make_experiment(self):
        if type(self.alphas) is list:
            alpha_values = [round(i, 2) for i in self.alphas]  # Class Proportion
        else:
            alpha_values = [round(x, 2) for x in np.linspace(0, 1, self.alphas)]  # Class Proportion

        test = pd.concat([self.train_test[1], self.train_test[3]], axis='columns')

        test_pos = test.loc[test['class'] == 1]  # seperating positive test examples
        test_neg = test.loc[test['class'] == 0]  # seperating negative test examples
        for thr in self.thr:
            for sample_size in self.batch_sizes:  # Varying test set sizes
                for alpha in alpha_values:  # Varying positive class distribution
                    for iteration in range(self.niterations):
                        pos_class_size = int(round(sample_size * alpha, 2))
                        neg_class_size = sample_size - pos_class_size

                        if pos_class_size is not sample_size:
                            positive_samples = test_pos.sample(int(pos_class_size), replace=False)
                        else:
                            positive_samples = test_neg.sample(frac=1, replace=False)

                        negative_samples = test_neg.sample(int(neg_class_size), replace=False)

                        test_sample = pd.concat([positive_samples, negative_samples])
                        testY = test_sample["class"]
                        testX = test_sample.drop(['class'], axis='columns')

                        # Counting num of actual positives in test sample
                        n_pos_test_sample = list(testY).count(1)

                        # actual pos class prevalence in generated sample
                        calc_prop_pos_class = round(n_pos_test_sample / len(test_sample), 2)

                        for iquantifier in self.quantifiers:
                            # .............Calling of Methods.................
                            pred_pos_prop = self.apply_quantifier(quantifier=iquantifier, clf=self.clf,
                                                                  thr=thr,
                                                                  measure=self.measure[iteration],
                                                                  train=[self.train_test[0], self.train_test[2]],
                                                                  test=testX)
                            pred_pos_prop = round(pred_pos_prop[1], 2)  # Getting only the positive proportion

                            # absolute error
                            abs_error = round(abs(calc_prop_pos_class - pred_pos_prop), 2)
                            result = {'name':self.name, 'sample': iteration + 1, 'Test_size': sample_size, 'alpha': alpha,
                                      'actual_prop': calc_prop_pos_class, 'pred_prop': pred_pos_prop,
                                      'abs_error': abs_error, 'quantifier': iquantifier, 'threshold': thr}
                            result = pd.DataFrame([result])

                            self.table = pd.concat([None if self.table.empty else self.table, result], ignore_index=True)

    def return_table(self):
        return self.table


class MakeExperiments:
    def __init__(self, datasets_folder, path_experiment, niterations, batch_sizes, alphas, clf, thr):
        f_names = []
        data_l = []
        for file in os.listdir(datasets_folder):
            f_names.append(os.path.splitext(f"{file}")[0])
            data_l.append(pd.read_csv(f"{datasets_folder}\\{file}"))

        df = pd.DataFrame(columns=["name", "dataset"])
        df["name"] = f_names
        df["dataset"] = data_l

        self.datasets = df
        self.datasets_folder = datasets_folder
        self.path_experiment = path_experiment
        self.niterations = niterations
        self.batch_sizes = batch_sizes
        self.alphas = alphas
        self.clf = clf
        self.thr = thr
        self.final = pd.DataFrame()


    def make_experiments(self):
        exists = os.path.exists(f"{self.path_experiment}.csv")
        if exists:
            path = f"{self.path_experiment}exp.csv"
        else:
            path = f"{self.path_experiment}.csv"

        for i, data in enumerate(self.datasets["dataset"]):

            types = data.dtypes
            label_encoder = LabelEncoder()

            for i in range(len(types)):
                if (data.dtypes[i] == "object"):
                    labels = label_encoder.fit_transform(data.iloc[:, i])
                    data[data.columns[i]] = labels

            popColumn = data.pop("class")
            data = pd.concat([data, popColumn], axis=1)

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            train_test = [X_train, X_test, y_train, y_test]

            #                        train_test, niterations, batch_sizes, alphas, clf, thr
            experiment = Experiments(train_test,
                                     self.datasets["name"][i],
                                     self.niterations,
                                     self.batch_sizes,
                                     self.alphas,
                                     self.clf,
                                     self.thr)
            experiment.make_experiment()

            self.final = pd.concat([self.final if not self.final.empty else None, experiment.return_table()])

        self.final.to_csv(path, index=False)



#Example
if __name__ == '__main__':

    #p = "C:\\Users\Luiz Fernando\\JupyterFiles\\Quantifier-project\\Quantifiers\\"
    p = ""

    clf = RandomForestClassifier(n_estimators=200)

    exp = MakeExperiments(datasets_folder=f"{p}data",
                          path_experiment=f"{p}experiments\\experiments",
                          niterations=3,
                          batch_sizes=range(10, 100, 10),
                          alphas=[0.2, 0.5, 0.8],
                          clf=clf,
                          thr=[0.2, 0.5, 0.8])
    exp.make_experiments()
