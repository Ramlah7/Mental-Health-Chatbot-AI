import re

def detect_intent(text: str) -> str:
    text = text.lower()

    INTENT_PATTERNS = {
        "trivial_crisis": [
            r"spilled", r"died", r"lost", r"missed", r"accident", r"printer", r"coffee", r"phone died",
            r"stain", r"broken", r"flat tire", r"traffic", r"late", r"forgot", r"mess up", r"glitch",
            r"internet down", r"power outage", r"my car won't start", r"ran out of", r"small problem",
            r"minor issue", r"oopsie", r"didn't work", r"computer crashed", r"burnt the toast",
            r"missed the bus", r"dog ate my homework", r"lost my keys",r"not working"
        ],
        "social_awkward": [
            r"bless you", r"people-watching", r"wave back", r"said too early", r"awkward", r"uncomfortable",
            r"nervous", r"embarrassing", r"oops", r"my bad", r"foot in mouth", r"said the wrong thing",
            r"silence", r"cringey", r"weird stare", r"didn't know what to say", r"social blunder",
            r"fumbled words", r"misunderstanding", r"pretend to be busy", r"avoid eye contact",
            r"felt out of place", r"didn't fit in", r"said something dumb", r"blushed", r"stammered",r"blunder"
        ],
        "overwhelmed": [
            r"stuck", r"can't sleep", r"failed", r"overthinking", r"stressed", r"racing", r"scrolling",
            r"too much", r"deadline", r"pressure", r"burnout", r"exhausted", r"can't cope", r"swamped",
            r"drowning", r"panicking", r"anxious", r"mind won't stop", r"too many tasks", r"can't focus",
            r"mental block", r"feeling drained", r"on the verge", r"heavy burden", r"lost control",
            r"pulled in too many directions", r"can't handle", r"everything at once"
        ],
        "achievement": [
            r"presentation", r"proud", r"good day", r"perfect hair", r"stood up for myself", r"succeeded",
            r"won", r"promotion", r"accolade", r"milestone", r"celebrate", r"accomplished", r"mastered",
            r"excelled", r"triumph", r"victory", r"reached my goal", r"nailed it", r"crushed it",
            r"personal best", r"got it right", r"worked out perfectly", r"well done", r"top score",
            r"reached a new level", r"impressed myself", r"finally did it", r"earned it"
        ],
        "rant": [
            r"why", r"what’s the point", r"hate", r"annoy", r"worst", r"sucks", r"ridiculous", r"infuriating",
            r"can't stand", r"ugh", r"seriously", r"makes me mad", r"frustrating", r"outrageous",
            r"unbelievable", r"don't get it", r"absurd", r"terrible", r"nightmare", r"what a mess",
            r"grinds my gears", r"it's unfair", r"stupid", r"pointless", r"waste of time", r"get off my chest",
            r"venting", r"this is ridiculous"
        ],
        "sadness": [
            r"cry", r"hurt", r"sad", r"😭", r"upset", r"empty", r"alone", r"depressed", r"grief",
            r"heartbroken", r"lonely", r"down", r"tearful", r"unhappy", r"misery", r"despair",
            r"mourn", r"ache", r"sorrow", r"heavy heart", r"feeling low", r"can't stop crying",
            r"gloomy", r"somber", r"melancholy", r"lost hope", r"broken spirit", r"can't find joy",
            r"isolated"
        ],
        "introvert": [
            r"alone time", r"recharge", r"quiet", r"prefer staying in", r"social battery",
            r"drained by people", r"peaceful night", r"small groups", r"need my space", r"prefer solitude",
            r"not feeling social", r"homebody", r"low-key", r"reflecting", r"inner world",
            r"too much noise", r"avoid crowds", r"just me", r"my own thoughts", r"need a break from people",
            r"overstimulated", r"prefer a book", r"prefer a movie", r"deep conversations", r"one-on-one"
        ],
        "extrovert": [
            r"party", r"socialize", r"people person", r"energy from others", r"loud", r"gathering",
            r"big group", r"networking", r"excited to meet", r"love being around people", r"go out",
            r"make new friends", r"center of attention", r"energetic", r"buzzing", r"talkative",
            r"join a group", r"love to chat", r"always busy", r"social butterfly", r"thrive in company",
            r"can't wait to see everyone", r"group activities", r"get together", r"mingle"
        ],
        "gratitude": [
            r"thank you", r"appreciate", r"grateful", r"blessed", r"fortunate", r"kindness",
            r"good deed", r"helped me out", r"so nice", r"much obliged", r"indebted", r"can't thank you enough",
            r"really appreciate it", r"feeling thankful", r"heartfelt thanks", r"truly grateful",
            r"you're the best", r"what a lifesaver", r"so thoughtful", r"very kind", r"made my day",
            r"blessings", r"luckily", r"happy about", r"content", r"relief"
        ],
        "frustration": [
            r"frustrated", r"annoyed", r"irritated", r"exasperated", r"fed up", r"can't believe",
            r"this is ridiculous", r"grr", r"ugh", r"infuriating", r"so mad", r"it's driving me crazy",
            r"bothering me", r"aggravated", r"disappointed", r"what a pain", r"can't stand this",
            r"so annoying", r"beyond frustrated", r"pulling my hair out", r"seriously angry",
            r"not again", r"this is getting old", r"can't deal with this", r"making me crazy"
        ],
        "excitement": [
            r"excited", r"thrilled", r"can't wait", r"looking forward", r"awesome", r"amazing",
            r"fantastic", r"yay", r"woohoo", r"pumped", r"eager", r"hyped", r"stoked", r"overjoyed",
            r"ecstatic", r"buzzing with anticipation", r"so happy about", r"bring it on",
            r"giddy", r"can't believe it", r"this is great", r"brilliant", r"super excited",
            r"feeling good", r"upbeat", r"ready to go", r"let's do this"
        ],
        "confusion": [
            r"confused", r"don't understand", r"perplexed", r"huh", r"what happened", r"unclear",
            r"muddled", r"puzzled", r"baffled", r"bewildered", r"lost", r"doesn't make sense",
            r"scratching my head", r"not getting it", r"what's going on", r"fuzzy", r"blurry",
            r"can't figure out", r"mixed up", r"disoriented", r"jumbled", r"unfathomable",
            r"enigmatic", r"obscure", r"mystified", r"unsure"
        ],
        "curiosity": [
            r"wonder", r"curious", r"tell me more", r"how does it work", r"what if", r"interested",
            r"explain", r"fascinated", r"intrigued", r"want to know", r"what's the story", r"inquiring",
            r"pondering", r"questioning", r"seeking answers", r"what's behind it", r"unravel the mystery",
            r"discover", r"explore", r"learn about", r"enlighten me", r"got a question",
            r"what about", r"tell me all", r"eager to learn", r"new information"
        ]
    }

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return "neutral"