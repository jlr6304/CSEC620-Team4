import numpy as np # use of logarithm

"""
Preprocess of a dataset in TF-IDF in order to apply Naive Bayes classifier
    parameters
        training_set: unpreprocessed (tokenized) train set of SMSs
    returns
        prob_dictionnary: probabilities for each word to be a spam or a ham
        number_of_ham_SMSs
        number_of_spam_SMSs
"""
def preprocessor(training_set):
    #create two dictionaries of unique words, one for both ham and spam   
    ham_words={}
    spam_words={}
    
    number_of_ham_SMSs=0
    number_of_spam_SMSs=0

    #create a dictonary containing the two dictionaries of words
    prob_dictionnary = {'ham':ham_words,'spam':spam_words}
    
    #add 1 for all of the words to fix any issue where word doesn't appear in one of collection of SMSs
    a=1
    for SMS in training_set:
        for word_position in range(1, len(SMS)):
            current_word = SMS[word_position]
            if current_word not in ham_words:
                ham_words.update({current_word: a})
            if current_word not in spam_words:
                spam_words.update({current_word: a})

    #start adding 1 for each occurence depending if it is ham or spam
    for SMS in training_set:
        #if the first word is ham 
        if SMS[0] == 'ham':
            number_of_ham_SMSs+=1
            #enumerate through each word after the first inital
            for word_position in range(1,len(SMS)):
                current_word=SMS[word_position]
                if current_word in ham_words:
                    #add 1 to the occurence of the current word for ham
                    ham_words.update({current_word:((int)(ham_words[current_word])+1)})
                else:
                    print("this should not run")
                    ham_words.update({current_word:1})
        
        #if the first word is spam 
        if SMS[0] == 'spam':
            number_of_spam_SMSs += 1
            #enumerate through each word after the first inital
            for word_position in range(1, len(SMS)):
                current_word = SMS[word_position]
                if current_word in spam_words:
                    #add 1 to the occurence of the current word for spam
                    spam_words.update({current_word: ((int)(spam_words[current_word]) + 1)})
                else:
                    print("this should not run")
                    spam_words.update({current_word: 1})
    #count the nubmer of non-unqiue words by adding all ocurrences, separately for ham and spam
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
    print("number of unique ham words:", len(ham_words))
    print("number of unique spam words:", len(spam_words))
    
    for word in ham_words:
        #divide the occurence of the word by the total occurence of all words in ham SMSs
        ham_words.update({word: ((ham_words[word])/number_of_ham_words)})
    
    for word in spam_words:
        # divide the occurence of the word by the total occurence of all words in spam SMSs
        spam_words.update({word: ((spam_words[word])/number_of_spam_words )})

    return prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs

"""
Function that classifies a new SMS based on the probability dictionnary
    parameters:
        test_SMS: message to classify
        prob_dictionnary: probabilities for each word to be a spam or a ham
        number_of_ham_SMSs
        number_of_spam_SMSs
    returns:
        p_label: the predicted label for the test_SMS
"""
def classify(test_SMS,prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs):

    #find the probibility that it is ham
    ham_probs=prob_dictionnary['ham']
    #create intiail guess to be used by dividing the nubmer of ham SMSs by the total SMSs
    ham_initial_guess=number_of_ham_SMSs/(number_of_ham_SMSs+number_of_spam_SMSs)
    word_probs=[]
    #enumerate through each word after intial word 
    for word_position in range(1, len(test_SMS)):
        #if word is in the list of ham words then proceed 
        if test_SMS[word_position] in ham_probs:
            #add the probability of the word being in ham SMS to the array of all of the words probability of the test SMS 
            word_probs.append(ham_probs[test_SMS[word_position]])
    
    prob=0
    #multiply all the probabilities of the words being in ham SMS to get the overall probability of the message being ham 
    for word_prob in word_probs:
        prob += np.log(word_prob)
    ham_prob= np.log(ham_initial_guess) + prob

    #find the probability that it is spam
    spam_probs = prob_dictionnary['spam']
    spam_initial_guess =  number_of_spam_SMSs / (number_of_ham_SMSs + number_of_spam_SMSs)
    word_probs = []
    #enumerate through each word after initial word 
    for word_position in range(1, len(test_SMS)):
        #if word is in the list of spam words than proceed 
        if test_SMS[word_position] in ham_probs:
            #add the probability of the word being in spam SMS to the array of all of the words probability of the test SMS 
            word_probs.append(spam_probs[test_SMS[word_position]])
    
    prob=0
    #multiply all the probilities of the words being in spam SMS to get the overall probility of the message being spam 
    for word_prob in word_probs:
        prob += np.log(word_prob)
    spam_prob = np.log(spam_initial_guess) + prob

    #if the probaility of being ham is greater than probability of being spam than return ham and otherwise return spam
    if(ham_prob>spam_prob):
        #print("ham")
        return "ham"
    else:
       # print("spam")
        return "spam"


"""
Function that runs Multinomial Naive Bayes classifier
    parameters
        training_set: [list]  unpreprocessed (tokenized) train set of SMS
        test_set: [list]  unpreprocessed (tokenized) test set of SMS
    returns:
        p_labels: predicted labels of test set
        t_labels: true labels of test set
"""
def run(training_set, test_set):
    # Preprocess the training set and return the dictionary containing ham and spam dictionaries, number of ham SMSs, and number of Spam SMSs
    prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs = preprocessor(training_set)
    p_labels=[]
    t_labels=[]
    # Enumerate through each SMS of the test set
    for SMS in test_set:
        #predict the classification of the test SMS
        prediction=classify(SMS,prob_dictionnary,number_of_ham_SMSs,number_of_spam_SMSs)
        
        actual=SMS[0]
        #append the predicted and actual labels to their respected arrays
        p_labels.append(prediction)
        t_labels.append(actual)

    # --- Print all the predicted labels and the related true labels
    # n = len(test_set)
    # comp = [(p_labels[i], t_labels[i]) for i in range(n)]    
    # for c in comp:
    #     print(c)

    return p_labels, t_labels # arrays of predicted and actual labels
