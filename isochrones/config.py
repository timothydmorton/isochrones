import os

ISOCHRONES = os.getenv('ISOCHRONES',
                       os.path.expanduser(os.path.join('~','.isochrones')))

