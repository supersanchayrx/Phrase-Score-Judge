import PreProcessRefAudio
import PreProcessCandidates
import ProcessAudioArray
import dtw
import json
import numpy as np

#parameters
maxKeyOffset = 400
phraseVsKeyRatio = 0.75 #keep b/w 0->1
minVoicedFrames = 10    # min frames per phrase
snrThreshold = 1.75

RefFeatureData = PreProcessRefAudio.PreProcessRef()
CandidatesFeatureData = PreProcessCandidates.ProcessAudioFiles()


phraseResults = []    # phrase results (per-track with nested phrases)
trackScores = []      #ranker

#testing with an audio
for candidateIndex in range(len(CandidatesFeatureData)):
    candidatePhrases = CandidatesFeatureData[candidateIndex]
    snr = candidatePhrases[0]["snr"] if len(candidatePhrases)>0 else None

    if len(candidatePhrases) == 0:
        trackId = f"candidate_{candidateIndex+1}"
        trackEntry = {"track_id": trackId, "snr": None, "phrases": []}
        trackEntry["phrases"].append({
            "phrase_number": 1,
            "status": "skipped",
            "phrase_score": None,
            "label": "silent_track",
            "reason": "no voice activity detected in track"
        })
        phraseResults.append(trackEntry)
        trackScores.append({"track_id": trackId, "aggregate_score": None})
        print(f"Candidate {candidateIndex+1} skipped (no voiced phrases)")
        continue

    audioFileName = candidatePhrases[0]["FileName"]
    trackEntry = {"track_id": audioFileName, "snr": round(snr, 4) if snr is not None else None, "phrases": []}
    scoredPhraseScores = []

    for phraseIndex in range(len(RefFeatureData)):
        refPhrase = RefFeatureData[phraseIndex]
        phraseNum = phraseIndex + 1

        # phrase extraction failed cuz of less phrases spoken by candidate or shorter audio
        if phraseIndex >= len(candidatePhrases):
            trackEntry["phrases"].append({
                "phrase_number": phraseNum,
                "status": "skipped",
                "phrase_score": None,
                "label": "missing_phrase",
                "reason": "phrase not found in candidate audio"
            })
            continue

        candPhrase = candidatePhrases[phraseIndex]

        # if the audio len is fine but they havent spoken anything it means its noise
        candCents = candPhrase["cents"]
        voicedCount = int(np.sum(~np.isnan(candCents)))

        if voicedCount < minVoicedFrames:
            trackEntry["phrases"].append({
                "phrase_number": phraseNum,
                "status": "skipped",
                "phrase_score": None,
                "label": "insufficient_voiced",
                "reason": f"only {voicedCount} voiced frames (min: {minVoicedFrames})"
            })
            continue

        # dtw
        dtwScore = dtw.calcDTW(refPhrase, candPhrase)

        if dtwScore is None:
            trackEntry["phrases"].append({
                "phrase_number": phraseNum,
                "status": "skipped",
                "phrase_score": None,
                "label": "pitch_extraction_failed",
                "reason": "DTW could not be computed (empty voiced contour)"
            })
            continue

        # normalization of melody happened b4 so we deduct it here 
        refMedianCents = float(refPhrase["medianCents"])
        candMedianCents = float(candPhrase["medianCents"])
        keyOffset = abs(refMedianCents - candMedianCents)
        keyPenalty = max(0.0, 100.0 * (1.0 - keyOffset / maxKeyOffset))

        # fair score as said above (the normalization penalty)
        phraseScore = round(phraseVsKeyRatio * dtwScore + (1 - phraseVsKeyRatio) * keyPenalty, 2)

        #another penalty for noisy audio 
        snrPenalty = min(1.0,snr/snrThreshold)
        phraseScore = phraseScore*snrPenalty

        trackEntry["phrases"].append({
            "phrase_number": phraseNum,
            "status": "scored",
            "phrase_score": round(phraseScore, 2),
            "dtw_score": round(dtwScore, 2),
            "key_offset_cents": round(keyOffset, 2),
            "key_penalty": round(keyPenalty, 2),
            "snr_penalty": round(snrPenalty, 4),
            "label": None,
            "reason": None
        })
        scoredPhraseScores.append(phraseScore)

        print(f"  {audioFileName} | Phrase {phraseNum} | DTW: {dtwScore:.1f} | Key offset: {keyOffset:.0f}c | Final: {phraseScore}")

    # we get a simple mean of all 4 phrases to make a final score
    if len(scoredPhraseScores) > 0:
        aggregateScore = round(sum(scoredPhraseScores) / len(scoredPhraseScores), 2)
    else:
        aggregateScore = None

    phraseResults.append(trackEntry)
    trackScores.append({"track_id": audioFileName, "aggregate_score": aggregateScore})
    print(f"  => {audioFileName} aggregate: {aggregateScore}\n")

# rank system
scoredTracks = [t for t in trackScores if t["aggregate_score"] is not None]
unscoredTracks = [t for t in trackScores if t["aggregate_score"] is None]

scoredTracks.sort(key=lambda t: t["aggregate_score"], reverse=True)

ranking = []
for rank, track in enumerate(scoredTracks, start=1):
    ranking.append({
        "track_id": track["track_id"],
        "aggregate_score": track["aggregate_score"],
        "rank": rank
    })
for track in unscoredTracks:
    ranking.append({
        "track_id": track["track_id"],
        "aggregate_score": None,
        "rank": None
    })

#ai generated code for the json files
# ---- Write output files ----
import os
os.makedirs("Results", exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that aren't JSON serializable."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open("Results/results.json", "w") as f:
    json.dump(phraseResults, f, indent=2, cls=NumpyEncoder)
    print("Wrote Results/results.json")

with open("Results/ranking.json", "w") as f:
    json.dump(ranking, f, indent=2, cls=NumpyEncoder)
    print("Wrote Results/ranking.json")

# print final ranking
print("\n=== Final Ranking ===")
for entry in ranking:
    rank = entry["rank"] if entry["rank"] is not None else "N/A"
    score = entry["aggregate_score"] if entry["aggregate_score"] is not None else "N/A"
    print(f"  #{rank}  {entry['track_id']}  (score: {score})")