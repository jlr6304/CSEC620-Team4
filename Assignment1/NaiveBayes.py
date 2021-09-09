



def preprocessor(training_set):
    ham_words={}
    spam_words={}
    number_of_ham_SMSs=0
    number_of_spam_SMSs=0
    prob_dictionnary = {'ham':ham_words,'spam':spam_words}
    #add 1 for all of the words
    for SMS in training_set:
        for word_position in range(1, len(SMS)):
            current_word = SMS[word_position]
            if current_word not in ham_words:
                ham_words.update({current_word: 1})
            if current_word not in spam_words:
                spam_words.update({current_word: 1})
    #start adding 1 for each occurance depending if it is ham or spam
    for SMS in training_set:
        if SMS[0] == 'ham':
            number_of_ham_SMSs+=1
            for word_position in range(1,len(SMS)):
                current_word=SMS[word_position]
                if current_word in ham_words:
                    #add 1 to the occurance of the current word for ham
                    ham_words.update({current_word:((int)(ham_words[current_word])+1)})
                else:
                    print("this should not run")
                    ham_words.update({current_word:1})
        if SMS[0] == 'spam':
            number_of_spam_SMSs += 1
            for word_position in range(1, len(SMS)):
                current_word = SMS[word_position]
                if current_word in spam_words:
                    #add 1 to the occurance of the current word for spam
                    spam_words.update({current_word: ((int)(spam_words[current_word]) + 1)})
                else:
                    print("this should not run")
                    spam_words.update({current_word: 1})
    number_of_ham_words=0
    for word in ham_words:
        number_of_ham_words+=ham_words[word]
    number_of_spam_words = 0
    for word in spam_words:
        number_of_spam_words += spam_words[word]
    print("number of ham SMSs:",number_of_ham_SMSs)
    print("number of spam SMSs:",number_of_spam_SMSs)
    print("number of ham words:",number_of_ham_words)
    print("number of spam words:", number_of_spam_words)
    for word in ham_words:
        #divide the occurance of the word by the total occurance of all words in ham SMSs
        ham_words.update({word: ((ham_words[current_word])/number_of_ham_words)})
    for word in spam_words:
        # divide the occurance of the word by the total occurance of all words in spam SMSs
        spam_words.update({word: ((spam_words[current_word])/number_of_spam_words )})

    #print(prob_dictionnary)
    return prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs

def classify(test_SMS,prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs):

    #find the probibility that it is ham
    ham_probs=prob_dictionnary['ham']
    ham_initial_guess=number_of_ham_SMSs/(number_of_ham_SMSs+number_of_spam_SMSs)
    word_probs=[]
    for word_position in range(1, len(test_SMS)):
        if test_SMS[word_position] in ham_probs:
            word_probs.append(ham_probs[test_SMS[word_position]])
    prob=1
    for word_prob in word_probs:
        prob=prob*word_prob
    ham_prob=ham_initial_guess*prob

    #find the probibility that it is spam
    spam_probs = prob_dictionnary['spam']
    spam_initial_guess =  number_of_spam_SMSs / (number_of_ham_SMSs + number_of_spam_SMSs)
    word_probs = []
    for word_position in range(1, len(test_SMS)):
        if test_SMS[word_position] in ham_probs:
            word_probs.append(spam_probs[test_SMS[word_position]])
    prob = 1
    for word_prob in word_probs:
        prob = prob * word_prob
    spam_prob = spam_initial_guess * prob
    print("_______________")
    #print("ham:",ham_prob)
    #print("spam:",spam_prob)
    if(ham_prob>spam_prob):
        print("ham")
        return "ham"
    else:
        print("spam")
        return "spam"



def run(training_set, test_set):
    prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs = preprocessor(training_set)
    predicted_and_actual=[]
    for SMS in test_set:
        prediction=classify(SMS,prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs)
        actual=SMS[0]
        predicted_and_actual_tuple=(actual,prediction)
        predicted_and_actual.append(predicted_and_actual_tuple)
    p_labels=[]
    t_labels=[]
    for labels in predicted_and_actual:
        p_labels.append(labels[0])
        t_labels.append(labels[1])
    return p_labels, t_labels
