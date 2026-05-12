import numpy as np
import librosa
import soundfile as sf
import os
import json
import scipy.fftpack as fft
from scipy.signal import medfilt
import ProcessAudioArray
import VocalsExtractor


#parameters here
candidateTrackFolder = "assets/tracks"
audioFiles = librosa.util.find_files(candidateTrackFolder)
sampleRate = 22050
DebugSaveCleanAudio = False
denoisingStrength = 0.75 #keep b/w 0->1
denoisedCandidateAudios = []
processedCandidateAudios = []

def ProcessAudioFiles(ReadFiles=True):
    processedCandidateAudios.clear()

    # if ReadFiles:
    #     #finding audio files and storing it all in memory
    #     for audioFile in audioFiles:
    #         audioData,sample = librosa.load(audioFile, sr=sampleRate, mono=True)
    #         candidateAudioArray.append({
    #             "audio":audioData,
    #             "sampleRate":sample,
    #             "fileName":os.path.basename(audioFile)
    #             })

    #     print("Audio Files Loaded")

    # #stft on each of those audioFiles
    # for candidateAudio in candidateAudioArray:
    #     audioSample, audioPhase = librosa.magphase(librosa.stft(candidateAudio["audio"]))

    #     noisePower = np.mean(audioSample[:,:int(sampleRate*0.1)], axis=1)

    #     #audioNoiseMask = audioSample>noisePower[:,None]
    #     audioNoiseMask = np.where(audioSample>noisePower[:,None],1.0,(1-denoisingStrength))
    #     audioNoiseMask = audioNoiseMask.astype(float)
    #     audioNoiseMask = medfilt(audioNoiseMask,kernel_size=(1,5))

    #     cleanAudioData = audioSample*audioNoiseMask
    #     cleanAudio = librosa.istft(cleanAudioData*audioPhase)

    #     denoisedCandidateAudios.append(cleanAudio)

    #     if DebugSaveCleanAudio:
    #         outputPath = os.path.join(candidateTrackFolder,candidateAudio["fileName"].replace(".wav","_clean.wav"))
    #         sf.write(outputPath,cleanAudio,sampleRate)
    #         print(f"Audio File Created {outputPath}")
        
    #     processedCandidateAudio.append(ProcessAudioArray.SwiftProcessMethod(cleanAudio))

    for audioFile in audioFiles:
        audioFileName = os.path.basename(audioFile)
        vocalsOnlyCandidate, snr = VocalsExtractor.extractVocals(audioFile, sampleRate)
        processedCandidateAudio = ProcessAudioArray.SwiftProcessMethod(vocalsOnlyCandidate)

        cents = processedCandidateAudio["cents"]
        voiced = processedCandidateAudio["voiced"]
        confidence = processedCandidateAudio["confidence"]
        medianCents = float(np.nanmedian(cents))
        normalizedCents = cents - medianCents

        candidateData = [{
            "FileName": audioFileName,
            "phraseNumber": 1,
            "cents": normalizedCents,
            "snr": snr,
            "medianCents": medianCents,
            "voiced": voiced,
            "confidence": confidence
        }]

        processedCandidateAudios.append(candidateData)
        voiceStartTime = processedCandidateAudio["voiceStartTime"]
        print(f"Candidate Audio {audioFile} Processed and start time observed at {voiceStartTime}")

    return processedCandidateAudios

#trial = ProcessAudioFiles()