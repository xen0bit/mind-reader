import csv
from tqdm import tqdm
#matching function
def word2EEG(wordTimings, eegtime):
    for word in reversed(wordTimings):
        if(eegtime > word['timestamp']):
            return word['word']

#Construct an object from words csv
wordTimings = []
with open('out.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        temp = {'timestamp': float(row['timestamp']), 'word': row['word']}
        wordTimings.append(temp)

print(wordTimings)

#Construct list of unique words for labels
wordLabel = []
for word in wordTimings:
    wordLabel.append(word['word'])
#Set deduplicates, list converts it back to a form we can grab and index from below
wordLabel = list(set(wordLabel))
print(wordLabel)

#Construct matchign the words to EEG data
output = []
with open('EEG_recording.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader):
        eegtime = float(row['timestamps'])
        matchedWord = word2EEG(wordTimings, eegtime)
        if(matchedWord is not None):
            row['wordClass'] = wordLabel.index(matchedWord)
            #row['word'] = matchedWord
            output.append(row)


with open('training.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10', 'Right AUX', 'wordClass']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in output:
        writer.writerow(row)
