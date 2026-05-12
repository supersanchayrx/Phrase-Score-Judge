import librosa
import json
import matplotlib
import numpy as np
import os
from swift_f0 import *
import ProcessAudioArray
import VocalsExtractor

#Skipping the de noising pipeline for ref audio as it's already super clean

#defining params here
sampleRate = 22050
debugMode=False
detector = SwiftF0(fmin=46.875, fmax=2093.75, confidence_threshold=0.7)
# leftOffset = 500
# rightOffset = 500
RefAudioProcessedData = None

referenceAudioPhraseData = []
refAudioPath = "assets/reference/vocals.wav"

def PreProcessRef():
    refVocals, snr = VocalsExtractor.extractVocals(refAudioPath, sampleRate)

    RefAudioProcessedData = ProcessAudioArray.SwiftProcessMethod(refVocals)

    cents = RefAudioProcessedData["cents"]
    voiced = RefAudioProcessedData["voiced"]
    confidence = RefAudioProcessedData["confidence"]
    medianCents = float(np.nanmedian(cents))
    normalizedCents = cents - medianCents

    referenceAudioPhraseData.append({
        "FileName": os.path.basename(refAudioPath),
        "phraseNumber": 1,
        "cents": normalizedCents,
        "medianCents": medianCents,
        "voiced": voiced,
        "confidence": confidence
    })

    voiceStartTime = RefAudioProcessedData["voiceStartTime"]
    print(f"Reference Audio Processed and start time observed at {voiceStartTime}")

    return referenceAudioPhraseData

    
#print(timeWindows)

#slice audio windows usiung the timeframes 
# analysisData = []
# segmentCount = 1
# combinedSegments = []
# referenceInfo = []
# for times in timeWindows:
#     timeStart, timeEnd = times

#     sampleStart = int(timeStart*sampleRate/1000)
#     sampleEnd = int(timeEnd*sampleRate/1000)

#     segment = refAudio[sampleStart:sampleEnd]
#     combinedSegments.append(segment);

#     analysisResult = detector.detect_from_array(segment, sampleRate)

#     #extract info from result
#     f0 = np.asarray(analysisResult.pitch_hz,dtype = float)
#     confidence = np.asanyarray(analysisResult.confidence,dtype=float)
#     vad = np.asarray(analysisResult.voicing,dtype=bool)
#     frameTimes = np.asarray(analysisResult.timestamps, dtype=float)

#     f01 = np.where((f0 > 0) & vad, f0, np.nan)
#     cents = 1200 * np.log2(f01/440)

#     #building the obj 
#     infoData = {
#         "times": times,
#         "frameTimes":frameTimes,
#         "f0" : f0,
#         "cents" : cents,
#         "voiced": vad,
#         "confidence":confidence,
#         "medianCents": float(np.nanmedian(cents)),
#         "voiceRatio": float(np.mean(vad)),
#         "meanConfidence": float(np.mean(confidence))
#     }

#     referenceInfo.append(infoData)

#     plot_pitch(analysisResult, show=False, output_path=f"pitch{segmentCount}.jpg")
#     segmentCount = segmentCount+1


#combinedSegmentAnalysis = detector.detect_from_array(combinedSegments, sampleRate)
#plot_pitch(combinedSegmentAnalysis, show=False, output_path=f"pitchSegmentsCombined.jpg")



#sampleStart1 = int(timeWindows[0][0]*sampleStart/1000)
#sampleEnd1 = int(timeWindows[3][1]*sampleStart/1000)

# dirtyData = refAudio[int((timeWindows[0][0]-leftOffset)*sampleRate/1000):int((timeWindows[3][1]+rightOffset)*sampleRate/1000)]
# RefAudioProcessedData = ProcessAudioArray.SwiftProcessMethod(dirtyData)

# dirtyDataResult = detector.detect_from_array(dirtyData, sampleRate)
# print(type(dirtyDataResult),[a for a in dir(dirtyDataResult) if not a.startswith("_")])
# plot_pitch(dirtyDataResult, show=False, output_path=f"pitchSegmentsCombinedDirty.jpg")

#infoNumber = 1
#for info in referenceInfo:
#    print(f"info {infoNumber} -> {info}")
#    infoNumber+=1


# print("Reference Audio Processed")
    


