def levenshtein_similarity(s1, s2):
    """
    Compute the Levenshtein similarity between two strings.
    """
    m, n = len(s1), len(s2)
    d = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    distance = d[m][n]
    max_length = max(len(s1), len(s2))
    similarity = 1 - (distance / max_length)
    return similarity

def main():
    s1 = "king"
    s2 = "kind"
    print(levenshtein_similarity(s1, s2))

if __name__ == "__main__":
    main()
    
