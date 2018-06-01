import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

ISOCHRONES = os.getenv('ISOCHRONES',
                       os.path.expanduser(os.path.join('~','.isochrones')))

POLYCHORD = os.getenv('POLYCHORD',
                    os.path.expanduser(os.path.join('~','PolyChord')))
