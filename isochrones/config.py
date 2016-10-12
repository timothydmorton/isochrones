import os

ISOCHRONES = os.getenv('ISOCHRONES',
                       os.path.expanduser(os.path.join('~','.isochrones')))

if not os.path.exists(ISOCHRONES):
    os.makedirs(ISOCHRONES)

POLYCHORD = os.getenv('POLYCHORD',
                    os.path.expanduser(os.path.join('~','PolyChord')))
