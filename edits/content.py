"""Everything the reel says and shows: the photo order, labels, timing and the end card.

Durations are in beats of the music (128 BPM, one beat = 0.469 s), so every cut lands on the beat.
The cut is kept under 25 seconds: the 12 strongest costumes, a vote grid and the closing card.
"""

BRAND = {
    "name": "Tiny Stars Daycare",
    "phone": "(780) 230-1599",
    "web": "tinystars.ca",
    "email": "info@tinystars.ca",
    "address": "10629 West Side Drive, Grande Prairie, AB",
    "instagram": "@tinystarsgp",
}

TITLE = {"kicker": "Tiny Stars", "big": "Halloween", "sub": "Costume Parade", "cta": "Pick your favourite!"}

# the cold open before the music starts
COLD = {"line1": "Something spooky", "line2": "happened at..."}

# photo: file stem in work/jpg, label: costume name, beats: on-screen time,
# fx: the transition INTO this shot (lightning, glitch, ghost, flicker, fog, zoom)
SHOTS = [
    {"photo": "IMG_8302", "label": "Ghost Buster", "beats": 4, "fx": "lightning"},
    {"photo": "IMG_8187", "label": "Little Witch", "beats": 2, "fx": "glitch"},
    {"photo": "IMG_8030", "label": "Baby Chick", "beats": 2, "fx": "flicker"},
    # no face to blur (the mask covers it), so frame on the mask instead
    {"photo": "IMG_8386", "label": "Skull King", "beats": 2, "fx": "lightning", "frame_box": [1137, 882, 139, 179]},
    {"photo": "IMG_8113", "label": "Busy Bee", "beats": 2, "fx": "ghost"},
    {"photo": "IMG_8379", "label": "Web Slinger", "beats": 2, "fx": "glitch"},
    {"photo": "IMG_8309", "label": "Ice Princess", "beats": 2, "fx": "zoom"},
    {"photo": "IMG_8053", "label": "Tiny Skeleton", "beats": 2, "fx": "flicker"},
    {"photo": "IMG_8176", "label": "Baby Dragon", "beats": 2, "fx": "glitch"},
    {"photo": "IMG_8335", "label": "Rapunzel", "beats": 2, "fx": "ghost"},
    {"photo": "IMG_8022", "label": "Super Plumber", "beats": 2, "fx": "zoom"},
    # big kids: widen and lower the frame so both Ghostface kids on the floor fit
    {"photo": "IMG_8274", "label": "Spooky Squad", "beats": 4, "fx": "lightning",
     "zoom": 1.22, "dx": 0.05, "face_pos": 0.33},
]

# not in the 25-second cut (kept so the face pipeline still covers them)
FAMILY = {
    "line": "Even the grown-ups dressed up!",
    "shots": [
        {"photo": "IMG_8002", "beats": 4, "fx": "fog"},
        {"photo": "IMG_8252", "beats": 4, "fx": "ghost"},
    ],
}

# the numbered grid that asks for comments
VOTE = {"big": "Which one wins?", "sub": "Comment the number below", "beats": 4, "fx": "glitch"}

# the closing card on the haunted classroom
OUTRO = {
    "line1": "Thanks for",
    "line2": "Watching",
    "tag": "See you in our next adventure",
    "stay": "Stay connected with our little big family",
    "beats": 10,
    "fx": "fog",
}

# Photos looked at but not used (kept here so the choice is easy to revisit).
# The first ten were in the 45-second cut.
SPARES = ["IMG_8059", "IMG_8227", "IMG_8202", "IMG_8191", "IMG_8131", "IMG_8359", "IMG_8224", "IMG_8367",
          "IMG_8049", "IMG_8306",
          "IMG_8287", "IMG_8314", "IMG_8093", "IMG_8106", "IMG_8238",
          "IMG_8249", "IMG_8173", "IMG_8109", "IMG_8205", "IMG_8326", "IMG_8136", "IMG_8090"]
