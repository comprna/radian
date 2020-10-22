import csv
import ast

def main():
    with open('hek293-fold1.csv', "r") as f:
        i = 0
        max_len = 0
        while True:
            line = f.readline()
            i += 1
            # Move back to beginning of file if at end
            if line == "":
                print("End of file at line {0}".format(i))
                break

            signal, sequence = line.split('\t')
            sequence = ast.literal_eval(sequence)
            
            len_sequence = len(sequence)
            if len_sequence > max_len:
                max_len = len_sequence
        
        print("Maximum sequence length: {0}".format(max_len))

if __name__ == "__main__":
    main()