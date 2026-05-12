from audio_separator.separator import Separator
from pathlib import Path
import os
import numpy as np
import librosa

modelFolder = Path("models")
#seperator = Separator(model_file_dir=str(modelFolder))
cleanAudioWriteFolder = Path("assets/cleanAudioCache")
#cleanAudioWriteFolder = os.path.join("assets/tracks/","cleanAudioCache")
sr2 = 22050
useGpuAccln = False

seperator = Separator(model_file_dir=str(modelFolder), output_dir=str(cleanAudioWriteFolder), output_format="WAV")
seperator.load_model(model_filename="Kim_Vocal_2.onnx")

def extractVocals(candidateAudio,sampleRate):
    
    candidateAudioPath = Path(candidateAudio)

    cached = list(cleanAudioWriteFolder.glob(f"*{candidateAudioPath.stem.replace('__','_')}*(Vocals)*.wav"))

    instrumentalCached = list(cleanAudioWriteFolder.glob(f"*{candidateAudioPath.stem.replace('__','_')}*(Instrumental)*.wav"))

    if cached :
        vocalsData, _sampleRate = librosa.load(cached[0],sr=sampleRate, mono=True)
        print(f"Already found extracted vocals for {candidateAudio}")

        if instrumentalCached:
            instrumentalData, _ = librosa.load(instrumentalCached[0], sr=sampleRate, mono=True)
            vocal_rms = np.sqrt(np.mean(vocalsData ** 2))
            instrumental_rms = np.sqrt(np.mean(instrumentalData ** 2))
            snr = vocal_rms / (instrumental_rms + 1e-10)
        else:
            snr = None
        return vocalsData, snr
        #vocalsOnlyArray.append(vocalsData)
    
    print(f"No Cached Vocals found")
    outputFiles = seperator.separate(str(candidateAudio))
    vocalOutputFile = next(p for p in outputFiles if "Vocals" in Path(p).name)
    vocalPath = cleanAudioWriteFolder / vocalOutputFile
    vocalsData, _sampleRate = librosa.load(vocalPath, sr=sampleRate, mono=True)

    instrumentalOutputFile = next((p for p in outputFiles if "Instrumental" in Path(p).name), None)
    if instrumentalOutputFile:
        instrumentalPath = cleanAudioWriteFolder / instrumentalOutputFile
        instrumentalData, _ = librosa.load(instrumentalPath, sr=sampleRate, mono=True)
        vocal_rms = np.sqrt(np.mean(vocalsData ** 2))
        instrumental_rms = np.sqrt(np.mean(instrumentalData ** 2))
        snr = float(vocal_rms / (instrumental_rms + 1e-10))
    else:
        snr = None

    return vocalsData, snr
    #vocalsOnlyArray.append(vocalsData)


#testVocals = extractVocals("assets/tracks/2f26152c-be35-4c63-bf2f-e7dc0763e89a__ae2be6a7-3996-4405-bd2b-90d632a044a6.wav",sr2)

        
        
    


