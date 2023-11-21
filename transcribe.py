from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def transcribe_audio(audio_path):
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load and preprocess the audio
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Transcribe audio to text
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription


transcription_result = transcribe_audio("myvoice/download.flac")
print(transcription_result[0].lower())