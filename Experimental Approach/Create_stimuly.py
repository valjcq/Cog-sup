import pygame

def geometric_shapes():
    # Initialize variable for color
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    gray = (128, 128, 128)
    W = 800
    H = 600
    screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
    pygame.display.set_caption('square')
    screen.fill(white)

    middle = (W/2, H/2)

    rec_width = 300
    rec_height = 100

    pygame.draw.rect(screen, red, (middle[0] - rec_width/2, middle[1] - rec_height/2, rec_width, rec_height))
    pygame.display.flip()

    quit = False
    while not quit:
        pygame.time.delay(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
    pygame.quit()

def circles():
    # Initialize variable for color
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    gray = (128, 128, 128)
    W = 800
    H = 600
    screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
    pygame.display.set_caption('square')
    screen.fill(white)

    middle = (W/2, H/2)

    circle_radius = 40
    pygame.draw.circle(screen, green, middle, circle_radius - 32)
    pygame.draw.circle(screen, blue, (middle[0] - (circle_radius/2 + 40), middle[1]), circle_radius)
    pygame.draw.circle(screen, red, (middle[0] + (circle_radius/2 + 40), middle[1]), circle_radius)
    pygame.display.flip()

    quit = False
    while not quit:
        pygame.time.delay(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
    pygame.quit()


def troxler():
    # Initialize variable for color
    colors = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'gray': (128, 128, 128)
    }
    lighter_colors = {}
    for key, value in colors.items():
        lighter_colors[key] = tuple(min(255, 180 + x) for x in value)
    print(lighter_colors)
    W = 800
    H = 600
    screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
    screen.fill(colors['white'])
    pygame.display.set_caption('square')

    middle = (W/2, H/2)

    circle_radius = 10
    space = 100
    pygame.draw.circle(screen, colors['black'], middle, 3)
    # green circle
    pygame.draw.circle(screen, lighter_colors['green'], ((middle[0] - space), middle[1]), circle_radius)
    pygame.draw.circle(screen, lighter_colors['green'], ((middle[0] + space), middle[1] + space), circle_radius)
    # blue circle
    pygame.draw.circle(screen, lighter_colors['blue'], ((middle[0] - space), middle[1] - space), circle_radius)
    pygame.draw.circle(screen, lighter_colors['blue'], ((middle[0] + space), middle[1]), circle_radius)
    # pink circle
    pygame.draw.circle(screen, lighter_colors['magenta'], (middle[0], middle[1] - space), circle_radius)
    pygame.draw.circle(screen, lighter_colors['magenta'], (middle[0] - space, middle[1] + space), circle_radius)
    # yellow circle
    pygame.draw.circle(screen, lighter_colors['yellow'], (middle[0] + space, middle[1] - space), circle_radius)
    pygame.draw.circle(screen, lighter_colors['yellow'], (middle[0], middle[1] + space), circle_radius)

    pygame.display.flip()

    quit = False
    while not quit:
        pygame.time.delay(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
    pygame.quit()


if __name__ == '__main__':
    troxler()
