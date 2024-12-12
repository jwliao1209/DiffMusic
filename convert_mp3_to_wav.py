from pydub import AudioSegment


input_mp3 = "data/samples/f06eb04e688f5acb43716b1d4eb09b5c.mp3"
output_wav = "data/samples/2.wav"
audio = AudioSegment.from_mp3(input_mp3)
audio.export(output_wav, format="wav")

print(f"Converted {input_mp3} to {output_wav}")
