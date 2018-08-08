from nltk.corpus import names
import nltk
import random
import pickle
male_names = [(name,'male') for name in names.words('male.txt')]
female_names = [(name,'female') for name in names.words('female.txt')]

labeled_names_all = male_names + female_names


random.shuffle(labeled_names_all)

dev_test_names = labeled_names_all[6001:7000]

def gender_features(name):
    return {"last_letter":name[-1],
            "second_last_letter" : name[-2]}

feature_sets = []
for (n,gender) in labeled_names_all:
    feature_sets.append((gender_features(n),gender))

training_set = feature_sets[:6000]
dev_set = feature_sets[6001:7000]
test_set = feature_sets[7000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

#Pickling
save_classifier = open("gender_classifier.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

#classifier = pickle.load(open("gender_classifier.pickle","rb"))

print("The accuracy of this classifier : ",nltk.classify.accuracy(classifier,dev_set)*100)

errors = []
for (name,tag) in dev_test_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((name,guess,tag))

print("Number of Errors in Development Set : ",len(errors))

print(classifier.show_most_informative_features(20))
for i in range(20):
    name = input("Enter the name to check gender : ")
    print(classifier.classify(gender_features(name)))

