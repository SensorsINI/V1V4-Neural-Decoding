import pickle


def extract_importance(model):
    se = model.net.se
    weights = se.y[:, :]
    weights = weights.cpu()
    return weights


def save_importances(importances):
    with open('importances.pkl', 'wb') as f:
        pickle.dump(importances, f)


def save_labels(importance_labels):
    with open('importance_labels.pkl', 'wb') as f:
        pickle.dump(importance_labels, f)


def save_predictions(importance_predictions):
    with open('importance_predictions.pkl', 'wb') as f:
        pickle.dump(importance_predictions, f)
