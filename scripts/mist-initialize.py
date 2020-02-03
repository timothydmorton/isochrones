from isochrones import get_ichrone

mist = get_ichrone("mist", bands="JHK")
mist.initialize()

tracks = get_ichrone("mist", bands="JHK", tracks=True)
tracks.initialize()
