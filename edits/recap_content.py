"""Everything "Recap of Last Week" says and shows: the song, the shots, their timing in beats, the words on screen.

Durations are in beats of the music (116 BPM, one beat = 0.517 s), so every cut lands on the beat. 48 beats = 24.8 s.

A shot: clip (a blurred working copy), start (first frame), beats, speed (1 = real time; a list of (beat, speed) ramps it),
zoom (start, end), center (x, y as fractions of the frame) or follow (a face track id, with follow_offset), mood (0 bright
day .. 1 dusk), steady (how hard to smooth the camera's shake), in (how the shot arrives), logo (False hides the corner logo),
cast (it ends frozen as a sticker; its camera (zoom, x, y) pulls back to a different framing for the card).
Costume names are playful descriptions rather than character names, as in the costume reel.
"""
# "A Cute Ghost Story" by MMAudio. Its band comes in at beat 16, so the film starts at beat 4 and the lift lands on film
# beat 12. The closing card (film beat 38) joins the song's last bars, so the film ends on the song's real last notes.
MUSIC = {"file": "cute_ghost_story_512522.mp3", "bpm": 115.97, "first_beat": 0.005, "start": 4, "lift": 12,
         "join": 38, "ending": 134}
BEAT_FRAMES = 60 / MUSIC["bpm"] * 30


def reach(start, frame, beats):
    """The speed that gets from frame `start` to `frame` in this many beats."""
    return (frame - start) / (beats * BEAT_FRAMES)


TITLE = {"big": "Recap", "size": 318, "y": 800, "dx": 44, "sub": "OF LAST WEEK", "sub_dy": 205,
         "kicker": "A TINY STARS PRODUCTION", "kick_dy": -235}

SHOTS = [
    # the haunted halls, at dusk, while the song is still quiet
    # (the title is framed high: a child is standing on a table at the bottom of this clip)
    {"clip": "IMG_7983", "start": 0, "beats": 4, "speed": 0.55, "zoom": (1.36, 1.28), "center": (0.5, 0.40), "mood": 0.75,
     "steady": 10},
    {"clip": "IMG_7977", "start": 0, "beats": 2, "speed": 1.0, "zoom": (1.03, 1.1), "center": (0.5, 0.46), "mood": 0.62,
     "steady": 10, "in": {"kind": "whip", "dir": 1}},
    {"clip": "IMG_7980", "start": 44, "beats": 2, "speed": 1.4, "zoom": (1.0, 1.06), "mood": 0.75, "steady": 8,
     "in": {"kind": "whip", "dir": -1}},
    {"clip": "IMG_7979", "start": 8, "beats": 2, "speed": 1.0, "zoom": (1.0, 1.08), "center": (0.5, 0.52), "mood": 0.75,
     "steady": 8, "in": {"kind": "whip", "dir": 1}},
    # the song stops for a beat here, picks up on the next, and the band comes in as the camera goes through the wreath
    {"clip": "IMG_7996", "start": 0, "beats": 2, "speed": [(0, 0.9), (1, 0.7), (2, 0.12)], "zoom": (1.0, 1.0), "mood": 0.55,
     "in": {"kind": "whip", "dir": -1},
     "through": {"time": 0.42, "x": 0.73, "y": 0.40, "rx": 0.135, "ry": 0.088, "scale": 10.0}},
    # the lights come up: the cast
    {"clip": "IMG_7994", "start": 0, "beats": 5, "speed": reach(0, 30, 3), "zoom": (1.3, 1.25), "follow": 1,
     "follow_offset": (0.0, 0.1), "cast": {"beats": 2, "frame": 30, "kicker": "STARRING", "name": "Tiny Tiger",
                                           "text_y": 345, "pivot_y": 1100}},
    {"clip": "IMG_7995", "start": 9, "beats": 3, "speed": reach(9, 24, 1), "zoom": (1.62, 1.55), "follow": 0,
     "follow_offset": (0.0, -0.01), "punch": 0.05,
     "cast": {"beats": 2, "frame": 24, "kicker": "CO-STARRING", "name": "Blue Monster", "text_y": 345, "pivot_y": 1100}},
    {"clip": "IMG_7997", "start": 0, "beats": 4, "speed": reach(0, 24, 2), "zoom": (1.42, 1.36), "follow": 1,
     "follow_offset": (0.0, 0.085), "punch": 0.05,
     "cast": {"beats": 2, "frame": 24, "kicker": "WITH", "name": "Baby Chick", "text_y": 345, "pivot_y": 1100}},
    {"clip": "IMG_7982", "start": 24, "beats": 4, "speed": reach(24, 48, 2), "zoom": (1.42, 1.36), "center": (0.57, 0.62),
     "logo": False, "punch": 0.05,  # framed tight: the same child is on a table at the top left; the card behind is blurred
     "cast": {"beats": 2, "frame": 48, "kicker": "AND INTRODUCING", "name": "Fire Chief & Little Dino", "size": 118,
              "text_y": 345, "pivot_y": 1200, "camera": (1.05, 0.524, 0.56)}},
    # the whole gang
    {"clip": "IMG_7987", "start": 22, "beats": 2, "speed": 1.2, "zoom": (1.16, 1.08), "center": (0.5, 0.55), "punch": 0.05,
     "steady": 8, "in": {"kind": "flash", "k": 0.45}},
    {"clip": "IMG_7986", "start": 48, "beats": 2, "speed": 1.2, "zoom": (1.36, 1.26), "center": (0.5, 0.68), "steady": 10,
     "in": {"kind": "whip", "dir": 1}},
    {"clip": "IMG_7998", "start": 0, "beats": 3, "speed": 0.8, "zoom": (1.08, 1.0), "steady": 8,
     "in": {"kind": "flash", "k": 0.45}},
]

# b0, b1: the beats between which the line is on screen
CAPTIONS = [
    {"text": "Last week|at Tiny Stars…", "b0": 4, "b1": 7, "y": 1290, "size": 96},
    {"text": "the halls|got haunted", "b0": 7, "b1": 10, "y": 1290, "size": 96},
    {"text": "and then…", "b0": 10, "b1": 12, "y": 1290, "size": 110},
    {"text": "the little monsters|showed up", "b0": 12.5, "b1": 15, "y": 1270, "size": 92, "max_w": 880},
    {"text": "They took over|every classroom", "b0": 28, "b1": 32, "y": 1270, "size": 92, "max_w": 880},
    {"text": "and stole|every heart", "b0": 32.3, "b1": 35, "y": 1320, "size": 96},
]

WALL = {
    "beats": 3,
    "tile": (344, 480),
    "gap": 14,
    "crop_zoom": 0.74,
    "start": (0.0, 1.5, 3.0),
    "columns": [
        ["IMG_8302", "IMG_8030", "IMG_8113", "IMG_8309", "IMG_8176", "IMG_8049"],
        ["IMG_8187", "IMG_8386", "IMG_8379", "IMG_8053", "IMG_8335", "IMG_8059"],
        ["IMG_8274", "IMG_8131", "IMG_8191", "IMG_8202", "IMG_8227", "IMG_8306"],
    ],
    "caption": {"text": "…and so many more", "y": 880, "size": 88},
}

# a curtain call: hop is the film beat each cut-out starts its jump, so it lands on one of the song's closing hits
# (film beats 41.58, 43.08, 44.07, 45.07); button is the song's very last hit, when the whole cast jumps
END = {
    "beats": 10,
    "line1": "Thanks for",
    "line2": "watching",
    "tag": "See you in our next adventure",
    "button": 45.56,
    "cast": [
        {"clip": "IMG_7994", "frame": 30, "height": 500, "x": 470, "bottom": 1918, "hop": 42.79},
        {"clip": "IMG_7997", "frame": 24, "height": 440, "x": 690, "bottom": 1915, "hop": 43.78},
        {"clip": "IMG_7982", "frame": 48, "height": 400, "x": 215, "bottom": 1915, "hop": 41.29},
        {"clip": "IMG_7995", "frame": 24, "height": 340, "x": 905, "bottom": 2000, "hop": 44.78},
    ],
}
