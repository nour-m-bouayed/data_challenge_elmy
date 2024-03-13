import src.metrics as metrics
import darts

def evaluate(model, features, target, indices=None):
    predictions = model.predict(features)
    if indices is not None:
        print(predictions.shape)
        print(target.shape)
        print(indices)
        predictions = predictions[indices]
        target = target.iloc[indices]
    return metrics.weighted_accuracy(target,predictions)


def evaluate_forecaster(model, features, target, indices=None):
    predictions = model.predict(len(features), future_covariates=darts.timeseries.TimeSeries.from_dataframe(features))
    predictions = predictions.pd_dataframe().values
    if indices is not None:
        predictions = predictions[indices]
        target = target.iloc[indices]
    return metrics.weighted_accuracy(target,predictions)
