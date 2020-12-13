import pygame as pg
import time


class Cursor(pg.Rect):
    def __init__(self):
        pg.Rect.__init__(self, 0, 0, 1, 1)

    def update(self):
        self.left, self.top = pg.mouse.get_pos()


class Button(pg.sprite.Sprite):
    def __init__(self, image1, image2, x=200, y=200):
        self.image_normal = image1
        self.image_selection = image2
        self.image_actual = self.image_normal
        self.rect = self.image_actual.get_rect()
        self.rect.left, self.rect.top = (x, y)

    def update(self, face, cursor):
        if cursor.colliderect(self.rect):
            self.image_actual = self.image_selection
        else:
            self.image_actual = self.image_normal

        face.blit(self.image_actual, self.rect)


class Blackboard:

    def __init__(self):
        self.drawing = False
        self.last_pos = None
        self.w = 20
        self.color = (0, 0, 0)
        self.screen = None
        self.bt_procces1 = pg.image.load("buttom_process.png")
        self.bt_procces1 = pg.transform.scale(self.bt_procces1, [100, 30])
        self.bt_procces2 = pg.image.load("button_prosess2.png")
        self.bt_procces2 = pg.transform.scale(self.bt_procces2, [100, 30])

        self.bt_reset1 = pg.image.load("reset.png")
        self.bt_reset1 = pg.transform.scale(self.bt_reset1, [100, 30])
        self.bt_reset2 = pg.image.load("reset2.png")
        self.bt_reset2 = pg.transform.scale(self.bt_reset2, [100, 30])

        self.boton1 = Button(self.bt_procces1, self.bt_procces2, 680, 50)
        self.boton2 = Button(self.bt_reset1, self.bt_reset2, 680, 100)

        self.result = pg.sprite.Sprite()
        self.cursor1 = Cursor()
        self.name = "image.png"

    def show_blackboard(self):
        pg.init()
        self.screen = pg.display.set_mode((800, 800))
        self.screen.fill((119, 119, 119))

        self.blackboard_draw = pg.Surface((500, 500))
        self.blackboard_draw.fill((255, 255, 255))
        self.screen.blit(self.blackboard_draw, (50, 50))

        self.mainloop()

    def draw(self, event):
        position = pg.mouse.get_pos()
        bool_x = True if position[0] < 550 and position[0] > 50 else False
        bool_y = True if position[1] < 550 and position[1] > 50 else False
        can_draw = bool_x and bool_y
        if event.type == pg.MOUSEMOTION:
            if (self.drawing):
                if not can_draw:
                    return
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

    def capture(self, display, name, pos, size):  # (pygame Surface, String, tuple, tuple)
        image = pg.Surface(size)  # Create image surface
        image.blit(display, (0, 0), (pos, size))  # Blit portion of the display to the image
        pg.image.save(image, name)  # Save the image to the disk

    def others_events(self, event, cursor):
        """for add events of others class"""
        pass

    def draw_text(self, string, container, font_render, color=(0, 0, 0)):
        text = font_render.render(string, 1, color)
        rect_text = text.get_rect()
        centroX = container.width / 2
        centroY = container.height / 2
        diferencia_x = centroX - rect_text.center[0]
        diferencia_y = centroY - rect_text.center[1]
        self.screen.blit(text, [container.left + diferencia_x, container.top + diferencia_y])

    def mainloop(self):
        loop = 1
        while loop:
            # checks every user interaction in this list
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    loop = 0
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_s:
                        self.name = "image%s.png" % time.strftime("%Y-%m-%d %H:%M:%S")
                        self.capture(self.screen, self.name, (50,50), (500,500))
                    if event.key == pg.K_r:
                        self.screen.fill((119, 119, 119))
                        self.screen.blit(self.blackboard_draw, (50, 50))
                self.draw(event)
                self.others_events(event, self.cursor1)
                self.cursor1.update()
                self.boton1.update(self.screen, self.cursor1)
                self.boton2.update(self.screen, self.cursor1)
            pg.display.flip()
        pg.quit()

