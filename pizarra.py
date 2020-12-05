import pygame as pg

# pip install pygame==2.0.0.dev10
 
"""
With this program you can draw on the 
screen with pygame
 
 
pythonprogramming.altervista.org
"""
 
def init():
    global screen
 
    pg.init()
    screen = pg.display.set_mode((400, 400))
    mainloop()
 
 
drawing = False
last_pos = None
w = 1
color = (255, 0, 255)
 
 
def draw(event):
    global drawing, last_pos, w
 
    if event.type == pg.MOUSEMOTION:
        if (drawing):
            mouse_position = pg.mouse.get_pos()
            if last_pos is not None:
                pg.draw.line(screen, color, last_pos, mouse_position, w)
            last_pos = mouse_position
    elif event.type == pg.MOUSEBUTTONUP:
        mouse_position = (0, 0)
        drawing = False
        last_pos = None
    elif event.type == pg.MOUSEBUTTONDOWN:
        drawing = True
 
 
def mainloop():
    global screen
 
    loop = 1
    while loop:
        # checks every user interaction in this list
        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = 0
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    pg.image.save(screen, "image.png")
            draw(event)
        pg.display.flip()
    pg.quit()
 
 
init()
