# audio.py

class Audio:
	def init_audio(self):
		if self.audio_on:
			self.background_music.set_volume(0.35)
			self.step_sound.set_volume(0.6)
			self.end_sound.set_volume(0.99)
		else: # Turn off audio
			self.background_music.set_volume(0)
			self.step_sound.set_volume(0)
			self.end_sound.set_volume(0)
