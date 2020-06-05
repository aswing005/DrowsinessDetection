import pygame
import threading
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")
def play():
    play.on=1
    while(play.on):
            pygame.mixer.music.play()
    return
def call():
    t=threading.Thread(target=play)
    t.start()
    return

def playalarm():
    pygame.mixer.music.play()