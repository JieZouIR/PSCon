# Function to calculate the Longest Common Subsequence (LCS) between two strings
def lcs(string, sub):
    # Swap strings if the first one is shorter than the second
    if(len(string)< len(sub)):
        sub, string = string, sub

    # Initialize a 2D array to store LCS lengths
    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    # Dynamic programming approach to find LCS
    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

# Function to calculate ROUGE-L score between a candidate sentence and multiple reference sentences
# beta: parameter to control the balance between precision and recall
def sentence_rouge(refs, candidate, beta = 1.2):
    prec = []  # List to store precision scores for each reference
    rec = []   # List to store recall scores for each reference

    for reference in refs:
        # compute the longest common subsequence
        lcs_score = lcs(reference, candidate)
        # Calculate precision: LCS length divided by candidate length
        if len(candidate)>0:
            prec.append(lcs_score /float(len(candidate)))
        # Calculate recall: LCS length divided by reference length
        if len(reference)>0:
            rec.append(lcs_score /float(len(reference)))

    # Get maximum precision and recall scores
    prec_max = max(prec)
    rec_max = max(rec)

    # Calculate F-measure using the beta parameter
    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max ) /float(rec_max + beta**2*prec_max)
    else:
        score = 0.0
    return score