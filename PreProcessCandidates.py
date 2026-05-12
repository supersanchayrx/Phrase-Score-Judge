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
candidateAudioArray = []
sampleRate = 22050
DebugSaveCleanAudio = False
denoisingStrength = 0.75 #keep b/w 0->1
denoisedCandidateAudios = []
processedCandidateAudios = []

processedCandidatePhraseData = []



with open("assets/reference/struct.json","r") as audioInfoData:
        data = json.load(audioInfoData)

        timeWindows = [(phraseData["start_time_ms"], phraseData["end_time_ms"]) for phraseData in data["phrases"]]

        phraseStartRef = timeWindows[0][0]

def ProcessAudioFiles(ReadFiles = True):
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
        vocalsOnlyCandidate, snr = VocalsExtractor.extractVocals(audioFile,sampleRate)
        processedCandidateAudio = ProcessAudioArray.SwiftProcessMethod(vocalsOnlyCandidate)
        voiceStartTime = processedCandidateAudio["voiceStartTime"]

        frameTimes = processedCandidateAudio["frameTimes"]

        processedCandidatesPhrases = []
        for phraseIndex,times in enumerate(timeWindows):
            timeStart, timeEnd = times

            if voiceStartTime is None:
                continue

            phraseStartTime = voiceStartTime  + (timeStart-phraseStartRef)/1000.0
            phraseEndTime = voiceStartTime  + (timeEnd-phraseStartRef)/1000.0

            frameIndices = np.where((frameTimes>=phraseStartTime) & (frameTimes<=phraseEndTime)) [0]

            phraseCents = processedCandidateAudio["cents"][frameIndices]
            phraseVad = processedCandidateAudio["voiced"][frameIndices]
            phraseConfidence = processedCandidateAudio["confidence"][frameIndices]
            medianCents = float(np.nanmedian(phraseCents))

            normalizedCents = phraseCents-np.nanmedian(phraseCents)

            processedCandidatesPhrases.append({
            "FileName" : audioFileName,
            "phraseNumber": phraseIndex + 1,
            "cents" : normalizedCents,
            "snr": snr,
            "medianCents":medianCents,
            "voiced":phraseVad,
            "confidence":phraseConfidence
            })

        processedCandidateAudios.append(processedCandidatesPhrases)
        print(f"Candidate Audio {audioFile} Processed and start time observed at {voiceStartTime}")
             
    return processedCandidateAudios

#trial = ProcessAudioFiles()