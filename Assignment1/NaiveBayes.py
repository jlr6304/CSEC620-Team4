def preprocessor():
    pass

def classify():
    return prediction


def run(training_set, test_set):
    prob_dictionnary = preprocessor(training_set)
    test_set, t_labels= preprocessor(test_set)
    
    p_labels = [classify(new_data, prob_dictionnary) for new_data in test_set]

    return p_labels, t_labels
