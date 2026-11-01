"""Everything the reel says and shows: the photo order, labels, timing and the end card.

Durations are in beats of the music (129.2 BPM, one beat = 0.464 s), so every cut lands on the beat.
"""

BRAND = {
    "name": "Tiny Stars Daycare",
    "phone": "(780) 230-1599",
    "web": "tinystars.ca",
    "address": "10629 West Side Drive, Grande Prairie, AB",
    "instagram": "@tinystarsgp",
}

TITLE = {"kicker": "Tiny Stars", "big": "Halloween", "sub": "Costume Parade", "cta": "Pick your favourite!"}

# photo: file stem in work/jpg, label: costume name, beats: on-screen time,
# focus: optional (x, y) crop centre as fractions of the photo (defaults to the faces / centre)
SHOTS = [
    {"photo": "IMG_8302", "label": "Ghost Buster", "beats": 4},
    {"photo": "IMG_8187", "label": "Little Witch", "beats": 2},
    {"photo": "IMG_8379", "label": "Web Slinger", "beats": 2},
    {"photo": "IMG_8030", "label": "Baby Chick", "beats": 4},
    {"photo": "IMG_8113", "label": "Busy Bee", "beats": 2},
    # no face to blur (the mask covers it), so frame on the mask instead
    {"photo": "IMG_8386", "label": "Skull King", "beats": 2, "frame_box": [1137, 882, 139, 179]},
    {"photo": "IMG_8309", "label": "Ice Princess", "beats": 4},
    {"photo": "IMG_8059", "label": "Construction Pup", "beats": 2},
    {"photo": "IMG_8053", "label": "Tiny Skeleton", "beats": 2},
    {"photo": "IMG_8227", "label": "Little Monster", "beats": 2},
    {"photo": "IMG_8022", "label": "Super Plumber", "beats": 2},
    {"photo": "IMG_8202", "label": "Little Piglet", "beats": 4},
    {"photo": "IMG_8176", "label": "Baby Dragon", "beats": 2},
    {"photo": "IMG_8335", "label": "Rapunzel", "beats": 2},
    {"photo": "IMG_8191", "label": "Cowboy", "beats": 2},
    {"photo": "IMG_8131", "label": "Firefighter", "beats": 2},
    {"photo": "IMG_8359", "label": "Kitty Cat", "beats": 2},
    {"photo": "IMG_8224", "label": "Sleeping Beauty", "beats": 2},
    {"photo": "IMG_8367", "label": "Midnight Hero", "beats": 2},
    {"photo": "IMG_8049", "label": "Blue Alien", "beats": 2},
    {"photo": "IMG_8306", "label": "Skeleton Crew", "beats": 4},
    # big kids: widen and lower the frame so both Ghostface kids on the floor fit
    {"photo": "IMG_8274", "label": "Spooky Squad", "beats": 4, "zoom": 1.22, "dx": 0.05, "face_pos": 0.33},
]

FAMILY = {
    "line": "Even the grown-ups dressed up!",
    "shots": [
        {"photo": "IMG_8002", "beats": 4},
        {"photo": "IMG_8252", "beats": 4},
    ],
}

END = {
    "big": "Happy Halloween!",
    "sub": "from all of us at Tiny Stars",
    "cta": "Comment your favourite number",
    "enrol": "Now enrolling - book a tour",
}

# Photos looked at but not used (kept here so the choice is easy to revisit)
SPARES = ["IMG_8287", "IMG_8314", "IMG_8093", "IMG_8106", "IMG_8238",
          "IMG_8249", "IMG_8173", "IMG_8109", "IMG_8205", "IMG_8326", "IMG_8136", "IMG_8090"]
