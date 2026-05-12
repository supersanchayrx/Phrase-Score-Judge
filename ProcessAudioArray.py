import librosa
import json
import matplotlib
import numpy as np
from swift_f0 import *

#sampleRate = 22050
detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.7)
#audioInfoData = []

rmsThreshold = 0.001
minConfidence  = 0.6  

def SwiftProcessMethod(audioData, sampleRate = 22050, detectionFrames = 5):
    print("processing audio data")

    analysisResult = detector.detect_from_array(audioData, sampleRate)

    f0 = np.asarray(analysisResult.pitch_hz,dtype = float)
    confidence = np.asanyarray(analysisResult.confidence,dtype=float)
    vad = np.asarray(analysisResult.voicing,dtype=bool)
    frameTimes = np.asarray(analysisResult.timestamps, dtype=float)
    voiceStartTime = None

    # energy check to filter out silent buffers and then a vad to find first audio activity 

    for i in range(len(vad) - detectionFrames+1):
        if vad[i:i+detectionFrames].all():
            windowConfidence = np.mean(confidence[i:i+detectionFrames])
            if windowConfidence < minConfidence:
                continue

            # check there is actualy any audio energy at this buffer
            startSample = int(frameTimes[i] * sampleRate)
            endSample   = int(frameTimes[min(i + detectionFrames, len(frameTimes) - 1)] * sampleRate)
            endSample   = min(endSample, len(audioData))
            if startSample < endSample:
                windowRms = np.sqrt(np.mean(audioData[startSample:endSample] ** 2))
                if windowRms < rmsThreshold:
                    continue

            voiceStartTime = float(frameTimes[i])
            break

    f01 = np.where((f0 > 0) & vad, f0, np.nan)
    cents = 1200 * np.log2(f01/440)


    infoData = {
        "voiceStartTime":voiceStartTime,
        "frameTimes":frameTimes,
        "f0" : f0,
        "cents" : cents,
        "voiced": vad,
        "confidence":confidence,
        "medianCents": float(np.nanmedian(cents)),
        "voiceRatio": float(np.mean(vad)),
        "meanConfidence": float(np.mean(confidence))
    }

    return infoData

    #audioInfoData.append(infoData)