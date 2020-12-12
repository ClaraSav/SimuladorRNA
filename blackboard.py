import pygame as pg
import time

class blackboard:

    def __init__(self):
        self.drawing = False
        self.last_pos = None
        self.w = 20
        self.color = (0, 0, 0)
        self.screen = None

    def show_blackboard(self):
        pg.init()
        self.screen = pg.display.set_mode((400, 400))
        self.screen.fill((255, 255, 255))
        self.mainloop()

    def draw(self, event):
        if event.type == pg.MOUSEMOTION:
            if (self.drawing):
                mouse_position = pg.mouse.get_pos()
                if self.last_pos is not None:
                    pg.draw.line(self.screen, self.color, self.last_pos, mouse_position, self.w)
                self.last_pos = mouse_position
        elif event.type == pg.MOUSEBUTTONUP:
            mouse_position = (0, 0)
            self.drawing = False
            self.last_pos = None
        elif event.type == pg.MOUSEBUTTONDOWN:
            self.drawing = True

    def mainloop(self):
        loop = 1
        while loop:
            # checks every user interaction in this list
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    loop = 0
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_s:
                        name = "image%s.png" % time.strftime("%Y-%m-%d %H:%M:%S")
                        pg.image.save(self.screen, name)
                    if event.key == pg.K_r:
                        self.screen.fill((255, 255, 255))
                self.draw(event)
            pg.display.flip()
        pg.quit()
