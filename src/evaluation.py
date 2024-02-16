import src.metrics as metrics

def evaluate(model, features, target):
    predictions = model.predict(features)
    return metrics.weighted_accuracy(predictions, target)

