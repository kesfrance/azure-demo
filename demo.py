import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class IrisClassifier:
    def __init__(self, model_specs):
        self.model_name = model_specs["model_name"]
        self.hyperparameters = model_specs["hyperparameters"]
        self.datapath = model_specs["dataset"]
        self.train_test_split = model_specs["train_test_split"]
        self.train_validate_split = model_specs["train_validate_split"]
        self.metrics = model_specs["model_metrics"]
        self.get_data_set()
    
    def run(self):
        test, full_training_data = self.data_preperation(
            self.train_test_split["test"], self.dataset)
        validate, train = self.data_preperation(
            self.train_validate_split["validate"], full_training_data)
        x_validate, y_validate = self.x_y_split(validate, "Species")
        x_train, y_train = self.x_y_split(train, "Species")
        x_test, y_test = self.x_y_split(test, "Species")
        clf = self.fit_classifier(x_train, y_train)
        validation_pred = self.score_data(x_validate, clf)
        validation_accuracy = self.evaluate_scores(
            y_validate, validation_pred)

        test_pred = self.score_data(x_test, clf)
        test_accuracy = self.evaluate_scores(y_test, test_pred)
        dic = {
            "validation_predictions":validation_pred,
            "test_predictions": test_pred,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy 
        }
        return dic
        
        
        
    def get_data_set(self):
        self.dataset = pd.read_csv(self.datapath)
        self.dataset.drop("Id", inplace=True, axis=1)

    def data_preperation(self, splits, input_data):
        sub_data_idx = []
        for species in input_data.Species.unique():
            one_species = input_data[input_data.Species == species]
            sample_points = int(splits * len(one_species))
            one_species_sample = one_species.sample(
                n=sample_points, random_state=10)
            sub_data_idx.extend(one_species_sample.index)

        sub_data_idx.sort()
        is_test_data = self.dataset.index.isin(sub_data_idx)
        test = self.dataset[is_test_data]
        train = self.dataset[~is_test_data]
        return test, train

    def x_y_split(self, dataset, target):
        features = [x for x in dataset.columns if x != target]
        return dataset[features].values, dataset[target].values
    
    def fit_classifier(self, x_train, y_train):
        clf = RandomForestClassifier(**self.hyperparameters)
        clf.fit(x_train, y_train)
        return clf

    @staticmethod
    def score_data(features, model):
        preds = model.predict(features)
        return preds
    
    @staticmethod
    def evaluate_scores(expected, predictions):
        return accuracy_score(expected, predictions)

if __name__ == "__main__":
    import json
    model_specs = json.load(open("model_specs.json"))
    iris = IrisClassifier(model_specs)
    outpput = iris.run()
    print(outpput)