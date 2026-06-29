"""Built-in example sets so the tool runs out-of-the-box. Bring your own via a CSV (text,label)."""

POS = [
    "An absolute masterpiece.", "I loved every minute.", "Brilliant and moving.",
    "A wonderful, heartwarming film.", "Fantastic performances throughout.", "Genuinely delightful.",
    "A triumph from start to finish.", "Beautifully made and deeply satisfying.",
    "One of the best I've seen.", "Charming, funny, and smart.", "Superb in every way.",
    "An exhilarating experience.", "Truly excellent.", "A joy to watch.", "Gorgeous and gripping.",
    "Outstanding work.",
]
NEG = [
    "A boring, pointless waste of time.", "I hated every minute.", "Dull and lifeless.",
    "A painful, tedious slog.", "Terrible performances throughout.", "Genuinely awful.",
    "A disaster from start to finish.", "Poorly made and deeply frustrating.",
    "One of the worst I've seen.", "Clumsy, unfunny, and dumb.", "Dreadful in every way.",
    "An exhausting experience.", "Truly dreadful.", "A chore to watch.", "Ugly and boring.",
    "Shoddy work.",
]

FORMAL = [
    "I would be most grateful for your prompt response.",
    "Please find the requested documents attached herewith.",
    "We regret to inform you of the schedule change.",
    "Kindly confirm your attendance at your earliest convenience.",
    "It is my pleasure to introduce our new initiative.",
    "Thank you for your continued cooperation in this matter.",
    "I am writing to formally request a leave of absence.",
    "We sincerely apologize for any inconvenience caused.",
    "Should you require further information, do not hesitate to ask.",
    "The committee has reviewed your application thoroughly.",
    "I trust this message finds you well.",
    "We look forward to a productive collaboration.",
    "Please be advised that the deadline has been extended.",
    "Your feedback would be greatly appreciated.",
    "Allow me to express my gratitude for your assistance.",
    "This serves as a formal notification of the decision.",
]
CASUAL = [
    "hey wanna grab food later lol",
    "omg you have to see this it's hilarious",
    "ugh i'm so done with today",
    "lemme know if you're free this weekend",
    "that movie was lowkey kinda fire ngl",
    "k sounds good see ya then",
    "bruh i totally forgot about that",
    "we should def hang out soon!!",
    "idk man it's whatever",
    "yo can you send me that thing",
    "haha no worries it's all good",
    "gonna crash early tonight, super tired",
    "this is the best snack ever fr",
    "wait what? that's so random",
    "thanks a ton you're the best :)",
    "nah i'm good but thanks tho",
]

RUDE = [
    "Shut up, nobody asked for your opinion.", "You're an idiot if you think that will work.",
    "Get lost, you're completely useless.", "Wow, what a stupid thing to say.",
    "I don't care about your pathetic excuses.", "You're wasting my time, as usual.",
    "Nobody likes you, just go away.", "That's the dumbest idea I've ever heard.",
    "Quit whining and figure it out yourself.", "You're hopeless and always will be.",
    "Mind your own business, loser.", "Honestly, you're a complete disappointment.",
    "Stop being so annoying for once.", "You clearly have no idea what you're doing.",
    "Ugh, why are you even here?", "Do everyone a favor and be quiet.",
]
POLITE = [
    "Thank you so much for your help today.", "I really appreciate you taking the time.",
    "Would you mind sharing your thoughts?", "That's a great point, thanks for raising it.",
    "Please let me know if I can assist.", "I'm grateful for your patience with this.",
    "It was lovely to hear from you.", "Of course, happy to help anytime.",
    "Thanks for the thoughtful feedback.", "I hope you have a wonderful day.",
    "That makes a lot of sense, thank you.", "I appreciate your understanding.",
    "Wonderful work, well done!", "Kindly let me know what you prefer.",
    "Thanks again, this was really helpful.", "It's a pleasure working with you.",
]

_SETS = {"sentiment": (POS, NEG), "formality": (FORMAL, CASUAL), "toxicity": (RUDE, POLITE)}


def examples_for(concept):
    return _SETS[concept]
