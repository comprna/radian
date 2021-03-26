import json

def generate_conditional_model():
    # Initialise 6-mer counts to 0
    k6mer_counts = initialise_6mer_counts()

    # Get 6-mer counts in CDHIT90 singleton group computed by Jellyfish
    with open("/home/alex/OneDrive/phd-project/k-mer-analysis/0_1_Jellyfish/singleton-sequences90-6mer-counts-dumps.txt") as f:
        for line in f:
            key = line.split(" ")[0].strip()
            val = line.split(" ")[1].strip()
            # Overwrite the initial entry
            k6mer_counts[key] = int(val)

    print(len(k6mer_counts.keys()))

    # Compute probability of each 6-mer, given the preceding 5-mer
    k6mer_probs = {}
    for k6mer, count in k6mer_counts.items():
        # Count how many times the 5-mer is observed
        k5mer = k6mer[:5]
        count_k5mer = k6mer_counts[k5mer + "A"] + \
                      k6mer_counts[k5mer + "C"] + \
                      k6mer_counts[k5mer + "G"] + \
                      k6mer_counts[k5mer + "T"]

        # Get the probability of this 6-mer
        k6mer_probs[k6mer] = count / count_k5mer

    print(len(k6mer_probs.keys()))

    for k6mer, prob in k6mer_probs.items():
        print("{0}:\t{1}".format(k6mer, prob))

    with open('/home/alex/Documents/rnabasecaller/6mer-cond-probs.json', 'w') as f:
        json.dump(k6mer_probs, f)

def generate_model():
    # Initialise 6-mer counts to 0
    k6mer_counts = initialise_6mer_counts()

    # Get 6-mer counts in CDHIT90 singleton group computed by Jellyfish
    with open("/home/alex/OneDrive/phd-project/k-mer-analysis/0_1_Jellyfish/singleton-sequences90-6mer-counts-dumps.txt") as f:
        for line in f:
            key = line.split(" ")[0].strip()
            val = line.split(" ")[1].strip()
            # Overwrite the initial entry
            k6mer_counts[key] = int(val)
    
    # Compute total number of 6-mer counts
    sum_counts = 0
    for count in k6mer_counts.values():
        sum_counts += count
    
    # Compute probability of each 6-mer
    k6mer_probs = {}
    for k6mer, count in k6mer_counts.items():
        k6mer_probs[k6mer] = count / sum_counts
    print(len(k6mer_probs.keys()))

    for k6mer, prob in k6mer_probs.items():
        print("{0}:\t{1}".format(k6mer, prob))

    with open('/home/150/as2781/rnabasecaller/6mer-probs.json', 'w') as f:
        json.dump(k6mer_probs, f)

def initialise_6mer_counts():
    alphabet = ["A","C","T","G"]
    k6mer_counts = {}
    for letter1 in alphabet:
        for letter2 in alphabet:
            for letter3 in alphabet:
                for letter4 in alphabet:
                    for letter5 in alphabet:
                        for letter6 in alphabet:
                            k6mer = letter1+letter2+letter3+letter4+letter5+letter6
                            k6mer_counts[k6mer] = 0
    return k6mer_counts

def main():
    generate_conditional_model()

# def main():
#     with open('/home/150/as2781/rnabasecaller/6mer-probs.json', 'r') as f:
#         k6mer_probs = json.load(f)

#     for k6mer, prob in k6mer_probs.items():
#         print("{0}:\t{1}".format(k6mer, prob))

class RnaModel:
    def __init__(self, model):
        self.model = model

    def get6merProb(self, n1, n2, n3, n4, n5, n6):
        k6mer = n1 + n2 + n3 + n4 + n5 + n6
        # print("K-mer: {0}".format(k6mer))
        return self.model[k6mer]


if __name__ == "__main__":
    main()