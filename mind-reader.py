import pygame
import time
import csv

outputData = []

pygame.font.init()
screen = pygame.display.set_mode((800,600))
background = (0, 0, 0)
screen.fill((background))
myfont = pygame.font.Font(None, 128)
f = open("excerpt1.txt","r")
for i in range(30):
    for line in f:
        for word in line.split():
            text = myfont.render(word,1,(255,255,255))
            text_rect = text.get_rect(center=(800/2, 600/2))
            screen.blit(text, text_rect)
            #Push word to outputData with timestamp
            temp = {"timestamp": time.time(), "word": word}
            outputData.append(temp)
            #Display word
            pygame.display.update()
            time.sleep(0.2)
            screen.fill((background))
            pygame.display.update()
    f.seek(0)
pygame.display.quit()

#Write to csv
with open('out.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'word']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in outputData:
        writer.writerow(row)