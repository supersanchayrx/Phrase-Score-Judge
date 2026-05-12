import numpy as np
import librosa

def calcDTW(refData, candData):

    # d
    refCents = refData["cents"]
    candCents = candData["cents"]

    refVoiced = refCents[~np.isnan(refCents)]

    candVoiced = candCents[~np.isnan(candCents)]

    M = len(refVoiced)
    N = len(candVoiced)

    if M == 0 or N == 0:
        return None

    # costmatrix
    cost = np.abs(refVoiced[:, None] - candVoiced[None, :])  

    
    D, warpPath = librosa.sequence.dtw(C=cost)

    

    normalizedDistance = D[-1, -1] / len(warpPath)

    maxReasonableDistance = 300
    score = max(0.0, 100.0 * (1.0 - normalizedDistance / maxReasonableDistance))

    return score