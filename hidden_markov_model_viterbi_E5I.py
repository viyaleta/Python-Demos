'''Hidden Markov Model Viterbi Algorithm
Information:    Viyaleta Peterson
                University of Bridgeport, Spring 2017

Purpose:        To find the most probable path of sequence of nucleotides through Exon, 5', or Intron states provided a set of probabilities
                of nucleotides in each state and a set of probabilities of the nucleotide moving from one state into another; and to
                calculate the total probability of such path using the Viterbi decoding algorithm.

Description:    The following program is a Viterbi algorithm implementation of the Hidden Markov Model (HMM) as it pertains to a directed,
                weighed graph of states and DNA nucleotides.
                The set of states includes Exon, 5', and Intron. States also include Start and End but because each only has one connected edge,
                can be disregarded until the probability of the most likely math is being calculated.
                Furthermore, each state except for Start and End have probabilities of nucleotides being in that state associated with them.
                The set of nucleotides is the same for each state: Adenine, Thymine, Cytosine, Guanine. Each nucleotide is represented by its
                first letter capitalized in the program. However, when inputing the sequence, user may choose to input lower case characters.
                The program uses user input to create dictionaries of states-to-state movement or nucleotide-in-state probabilities.
                The program generates two matrices:
                    probability matrix, created by using the Viterbi algorithm, and
                    trace matrix, created to indicate whether a state-to-state movment was done to arrive in the current probability matrix location.
                Finally, using the trace matrix, program traces back the most likely path and using the probability matrix, determines the
                probability of that path.

Algorithm:      Input:  N = sequence of nucleotides
                        P(n|s) = initial probability of nucleotide in state s
                        a(ij) = probabiility of moving from state i into state j
                Output: X = array of n states where each state is the most probable state of the i-th nucleotide in the sequence N
                    1.  M <- table of probabilities where # of rows <- # states and # columns <- length of sequence
                    2.  D <- table of directions where # of rows <- # states and # columns <- length of sequence
                    3.  s <- number of states
                    4.  n <- length of sequence
                    5.  For each row i from 1 to s
                    6.        For each column j from 1 to n
                    7.            If P(i-1|j) * a(jj) > P(i-1|j-1) * a(ij)
                    8.                 M(i,j) <- P(i|j) + P(i-1|j) * a(jj)
                    9.                 D(i,j) <- 0
                    10.           Else
                    11.                M(i,j) <- P(i-1|j-1) * a(ij)
                    12.                D(i,j) <- -1
                    13. this_state <- s
                    14. For each column i from n to 1
                    15.        path(i) <- this_state
                    16.        While this_state > 0
                    17              X(i) <- this_state
                    18.             this_state <- this_state + D(this_state,i)

Instructions:   To run the program,
                1.    locate and open hmm_viterbi.py;
                2.    follow the on-screen instructions, inputing algorithm probability parameters;
                        if errors are encountered, they will display on screen;
                3.    after prompt, enter a sequence of nucleotides;
                4.    program will calculate and output the most likely path and the probability of that path;
                5.    after prompt,
                        type "Y" if you wish to try another sequence using current probability paramteres;
                        type "N" if you wish to exit;

Example:        Prompt                                                              Input
                ----------------------------------------------------------------    ---------
        Input:  Enter the probability of starting in Exon state:                    >>> 1
                Enter the probability of switching from Exon state to 5' state:     >>> 0.1
                Enter the probability of switching from 5' state to Intron state:   >>> 0.5
                Enter the probability of moving from Intron to End state:           >>> 0.1
                Enter the probability of A in Exon state:                           >>> 0.25
                Enter the probability of T in Exon state:                           >>> 0.25
                Enter the probability of C in Exon state:                           >>> 0.25
                Enter the probability of G in Exon state:                           >>> 0.25
                Enter the probability of A in 5 Prime state:                        >>> 0.8
                Enter the probability of T in 5 Prime state:                        >>> 0.15
                Enter the probability of C in 5 Prime state:                        >>> 0.05
                Enter the probability of G in 5 Prime state:                        >>> 0
                Enter the probability of A in Intron state:                         >>> 0.4
                Enter the probability of T in Intron state:                         >>> 0.4
                Enter the probability of C in Intron state:                         >>> 0.1
                Enter the probability of G in Intron state:                         >>> 0.1

                Enter a sequence of nucleotides (using only first letter for each nucleotide):  >>> tatat
                Using Viterbi algorithm, the most probable state of TATAT is
                Exon, 5 Prime, Intron, Intron, Intron,
                with overall probability of 5.184000000000002e-05
                --------------------------------------------------
                                      T                     A                     T                     A                     T
                Exon                  0.25                  0.05625               0.01265625            0.0028476562500000004 0.0006407226562500001
                5 Prime               0                     0.020000000000000004  0.0015000000000000002 0.0010125000000000002 7.593750000000001e-05
                Intron                0                     0                     0.004000000000000001  0.0014400000000000003 0.0005184000000000001
                --------------------------------------------------

                Would you like to enter another sequence? (Y/N)                                 >>> y
                Enter a sequence of nucleotides (using only first letter for each nucleotide):  >>> aatgt
                Using Viterbi algorithm, the most probable state of AATGT is
                Exon, 5 Prime, Intron, Intron, Intron,
                with overall probability of 1.2960000000000005e-05
                --------------------------------------------------
                                      A                     A                     T                     G                     T
                Exon                  0.25                  0.05625               0.01265625            0.0028476562500000004 0.0006407226562500001
                5 Prime               0                     0.020000000000000004  0.0015000000000000002 0.0                   4.271484375e-05
                Intron                0                     0                     0.004000000000000001  0.0003600000000000001 0.00012960000000000003
                Using Viterbi algorithm, the most probable state of AATGT is
                Exon, 5 Prime, Intron, Intron, Intron,
                with overall probability of 1.2960000000000005e-05

                Would you like to enter another sequence? (Y/N)                                 >>> y
                Enter a sequence of nucleotides (using only first letter for each nucleotide):  >>> aga
                --> Using the Viterbi algorithm, no probable path exists for AGA

                Would you like to enter another sequence? (Y/N)                                 >>> n
'''
from re import match

class Viterbi:
    def __init__(self):
        """ Initializes probability dictionaries with validated user input.
            Loops to call method new_sequence() until user wishes to exit to re-use the same probabilities
        """
        self.states = ['Exon','5 Prime','Intron']
        self.state_change = {'Exon':{}, '5 Prime':{}, 'Intron':{}}  #initializes dictionary, keys = from-states, and values = {to-states: probabilities} for probabilities of switching from one state to another
        self.nucleotide_at_state={'Exon': {}, '5 Prime': {}, 'Intron': {}}  #initializes dictionary, keys = states, and values = {nucleotides: probabilities} for probabilities of nucleodies in each state

        # Short program introduction
        print('The following program finds the most likely progression of states Exon, 5\', and Intron for a sequence of nucleotides based on input probabilities of switching from one state to another and input probabilities of a nucleotide to be in a particular state.')
        print('The following represents the graph used in this program:\n',
        '''
START --> Exon -->  5'  --> Intron --> End
          ||||     ||||      ||||
          ATCG     ATCG      ATCG
        ''')
        print('-'*50)

        # User input prompt for from-state to-state probabilies
        self.state_change['Exon']['5 Prime'] = self.validate_probability_input('Enter the probability of switching from Exon state to 5\' state: ')  #probability Exon -> 5'
        self.state_change['5 Prime']['Intron'] = self.validate_probability_input('Enter the probability of switching from 5\' state to Intron state: ')  #probability 5' -> Intron
        self.prob_end = self.validate_probability_input('Enter the probability of moving from Intron to End state: ')  #probability Intron -> End
        self.state_change['Exon']['Exon'] = 1 - self.state_change['Exon']['5 Prime']  #probability of remaining in Exon state
        self.state_change['5 Prime']['5 Prime'] = 1 - self.state_change['5 Prime']['Intron']  #probability of remaining in 5' state
        self.state_change['Intron']['Intron'] = 1 - self.prob_end  #probability of remaining in Intron state

        # User input prompt for nucleotide-in-state probabilities
        for state in self.states:
            while True:  #breaks when nucleotide-in-state probabilities are validated via validate_probability_sum() method
                for nucleotide in ['A','T','C','G']:  #requests user input of probability for each nucleotide within the state
                    self.nucleotide_at_state[state][nucleotide] = self.validate_probability_input('Enter the probability of {} in {} state: '.format(nucleotide,state))
                if self.validate_probability_sum(state) != True:  #uses validate_probability_sum() method to check if sum of probabilities of all nucleotides within the each state is exactly 1
                    print('The sum of probabilities of nucleotides in {} are greater than 1. Please re-enter the probabilities.'.format(state))  #prints error if input is not validated
                else:  #breaks the while loop for this state if input is validated
                    break

        while True:
            self.new_sequence()  #calls method to prompt user to a enter a new sequence and validates the sequence
            repeat = input('\nWould you like to enter another sequence? (Y/N) ').upper()
            while repeat != 'Y' and repeat != 'N':  #as long as user input is invalid, prompt to re-enter input
                repeat = input('Please enter Y or N only. \nWould you like to enter another sequence? (Y/N) ').upper()
            if repeat == 'N':  #if user typed "n" or "N", exit the program
                break

    def validate_probability_input(self,prompt):
        """This method validates user input for state or nucleotide probabilities, making sure that input is positive and is not greater than 1 and input can be converted to a float number."""
        usr_input = ''  #initiates local string variable
        while True:  #breaks when user input is <= 1 and can be turned into a float number
            try:
                usr_input = float(input(prompt))  #attempts to convert input string into float
                if usr_input > 1: raise ArithmeticError  #raises exception if user enters probability > 1
                if usr_input < 0: raise ValueError  #raises exception if user enters negative probability
                break  #breaks if no errors
            except ArithmeticError:
                print('Error: Probability cannot be greater than 1!')  #error message if ArithmeticError raised, when usr_input > 1
            except:
                print('Error: Please enter a positive decimal or integer number.')  #error message if other exception, incl ValueError raised
        return usr_input  #returns prompt converted into a float number if no errors were raised


    def validate_probability_sum(self,state):
        """Validates user input to make sure that probabilities of nucleotides in each state add up to exactly 1.0"""
        state_total = 0  #initializes the sum of probabilities in the state
        for nucleotide in ['A','T','C','G']:
            state_total += self.nucleotide_at_state[state][nucleotide]  #for each nucleotide, adds the probability too state_total
        if state_total == 1:  #if the sum of probabilities of nucleotides in the state is exactly 1, then returns true (valid)
            return True


    def new_sequence(self):
        """ Prompts user to enter a nucleotide sequence and validates the characters in the sequence to make sure they match nucleotides,
            Calls methods build_matrix() and traceback() to find the most likely path of the sequence and its probability.
            """
        while True:  #breaks if characters in the sequence match capital or lowercase first letters of nucleotides (ex. A or a for Adenine)
            self.sequence = input('Enter a sequence of nucleotides (using only first letter for each nucleotide): ')
            self.sequence = self.sequence.upper()  #converts sequence into all caps
            if match('^[a,A,t,T,c,C,g,G]*$',self.sequence):  #using re library re.match, validates the sequence
                break
            print('You have entered an invalid sequence. Please try again using only A, T, C, or G characters.')  #error message if sequence is not validated above
        self.n = len(self.sequence)  #after validation, sets length of sequence
        self.build_matrix()  #using the Viterbi algorithm, buils a probability matrix
        self.traceback()  #using the Viterbi aglorithm, traces back the most probable path using the probability and trace matrices and calculates the total probability of that path


    def build_matrix(self):
        """Calculates the probability and trace matrices with Viterbi algorithm to be used in tracing back the most likely path"""
        self.matrix = [[0 for i in range(self.n)] for row_num in range(len(self.states))]  #initializes a probability matrix with rows representing states and columns representing sequence nucleotides
        self.trace = [[0 for i in range(self.n)] for row_num in range(len(self.states))]  #initializes a trace matrix with rows representing states and columns representing traceback direction with -1 as movement diagonally down and 0 as horizontal movement

        for row_num in range(len(self.states)):  #algorithm calculates one row at a time from top to bottom
            prev_state = self.states[row_num-1]  #sets previous state based on previous row
            state = self.states[row_num]  #sets current state based on row
            for i in range(self.n):  #self.n indicates the number of nucleotides in the user-defined sequence
                current_nucleotide = self.sequence[i]  #gets character of the i-th nucleotide in sequence
                current_probability = self.nucleotide_at_state[state][current_nucleotide]  #gets the probability of current nucleotide from nucleotide_at_state dictionary
                if row_num == 0:  #first state after start
                    if i == 0:  #probability in first row/column of the matrix is just the probability of nucleotide
                        self.matrix[row_num][i] = current_probability
                    else:  #probability of first row and consecutive columns is the probability of previous cell in the same column and probability of remaining the the same state
                        self.matrix[row_num][i] = current_probability * self.matrix[row_num][i-1] * self.state_change[state][state]
                        self.trace[0][i] = 0  #horizontal direction
                else:  #consecutive states
                    if (self.matrix[row_num-1][i-1] * self.state_change[prev_state][state]) > (self.matrix[row_num][i-1] * self.state_change[state][state]): #if diagonal move into this cell is greater than the horizonal move into this cell
                        max_incoming_probability = self.matrix[row_num-1][i-1] * self.state_change[prev_state][state]  #set the maximum as the probability of the diagonal move
                        direction = -1  #diagonal direction
                    else: #if probability of moving into this cell horizonally is greater than diagonally
                        max_incoming_probability = self.matrix[row_num][i-1] * self.state_change[state][state]  #set the maximum as the probability of the horizontal move
                        direction = 0  #horizontal direction

                    if (state == '5 Prime' and i == 0) or (state == 'Intron' and (i == 0 or i == 1)):
                        self.matrix[row_num][i] = 0  #fill first column of 5' and first two columns of Intron as 0s due to no vertical movement allowed
                    else:
                        self.matrix[row_num][i] = current_probability * max_incoming_probability  #for all other cells, cell value is probability of current cell times probability of either horizontal or vertical movement in
                    self.trace[row_num][i] = direction  #set corresponding cell in trace matrix to the movement direction


    def traceback(self):
        """Uses the trace matrix to find the most likely path and probability matrix to calculate the probability of that path."""
        path = [0 for i in range(self.n)]  #initializes path as a list
        row = len(self.trace) - 1  #initializes row as the last row in trace matrix
        for i in range(self.n-1, -1, -1):  #iterates backwards, from the last state to the first state
            path[i] = self.states[row]  #sets i-th item of path list as the state that represents the row of the current cell
            if row != 0:  #evaluates whether algorithm reached the first row
                row += self.trace[row][i]  #if algorithm has not reached first row, then add the direction (-1 for diagonal, 0 for horizontal) to the current row (ex: if row = 2 and direction into it was diagonal then new row = 2 + (-1) = 1, traceback moves up)

        path_prob = self.matrix[2][self.n-1] * self.prob_end  #calculates the path probability by taking the last row/column of probability matrix and multiplying by start and end contant probabilities

        self.output_result(path,path_prob)  #calls method to print the output

    def output_result(self,path,path_prob):
        """Generates on-screen print of the program output: path, probability of the path, and the probability matrix created by build_matrix method."""
        if path_prob == 0:  #if path_prob returned 0 as the probability, notifies user that sequence is improbable
            print('\n--> Using the Viterbi algorithm, no probable path exists for {}'.format(self.sequence))
        else:  #if path_prob is greater than 0, notifies user of the original sequence, path, and its probability, and outputs the probability matrix
            print('\nUsing Viterbi algorithm, the most probable state of {} is \n{} \nwith overall probability of {}'.format(self.sequence, path, path_prob))
            print('-'*50)
            header = ''
            for j in range(self.n):
                header += ('\t' + self.sequence[j])  #header displays sequence nucleotides input by the user
            print(header.expandtabs(24))
            for i in range(len(self.states)):
                row = self.states[i]  #each row is associated with each state
                for j in range(self.n):
                    row += ('\t' + str(self.matrix[i][j]))  #each row's cells are corresponding values from the probability matrix
                print(row.expandtabs(24))
            print('-'*50)

Viterbi()
