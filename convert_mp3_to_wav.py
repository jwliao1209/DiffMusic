from pydub import AudioSegment


input_mp3 = "data/samples/ec39eb2ef30e510d7d20db5146f3dfde.mp3"
output_wav = "data/samples/5.wav"
audio = AudioSegment.from_mp3(input_mp3)
audio.export(output_wav, format="wav")

print(f"Converted {input_mp3} to {output_wav}")
