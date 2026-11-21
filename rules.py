"""Keyword lists for rule-based mode (English + Turkish). Kept in one place so they are easy to tune."""
from __future__ import annotations

import re

SPAM = ["generator", "dm me", "selling account", "discord server", "sub to my", "youtube",
        "free coins", "linke tıkla", "ücretsiz", "wanna be friends", "http", ".example"]
FEATURE = ["please add", "add ", "would love", "suggestion", "option to", "can we get", "would be cool",
           "would be nice", "let us", "make tournaments", "ekleyin", "olsa güzel", "pls add"]
QUESTION_START = ["how ", "when ", "what ", "is there", "is cross", "can i ", "do ", "does ", "nasıl",
                  "ne zaman", "why can"]

# Bug signal words, grouped by the severity category they suggest.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "payment": ["charged", "paid", "payment", "purchase", "bought", "refund", "receipt", "order id",
                "subscription", "restore purchase", "vip", "satın", "ödedim", "para ", "abonelik", "iap"],
    "crash": ["crash", "closes", "closing", "won't start", "wont start", "cant even open", "çöküyor",
              "açılmıyor", "kapanıyor", "black screen"],
    "data_loss": ["lost my", "gone from", "disappeared", "vanished", "deleted", "missing", "kayboldu",
                  "silindi", "progress reset", "ate my"],
    "major_feature": ["stuck", "freez", "frozen", "donuyor", "softlock", "soft-lock", "disconnect",
                      "connection lost", "dropping connection", "kicks me", "kopuyor", "isnt updating",
                      "not updating", "not refreshing", "güncellenmiyor", "doesnt save", "hasn't changed",
                      "kaldım", "bitmiyor", "can't play", "cant play", "görünmüyor", "won't let", "wont let"],
    "minor": ["no sound", "silent", "audio", "ses ", "sound", "timer", "sayacı", "resets", "sıfırlan",
              "gelmiyor"],
    "cosmetic": ["flicker", "texture", "glitchy", "text goes outside", "taşıyor", "sığmıyor", "overflow",
                 "flashing"],
}
GENERIC_BUG = ["bug", "broken", "fix", "doesnt work", "doesn't work", "not working", "glitch", "bozuk",
               "hata", "something is wrong", "cant ", "can't ", "didn't", "gelmedi"]

# Turkish -> English hints so TF-IDF can match mixed-language duplicates.
TR_EN = {
    "çöküyor": "crash", "açılmıyor": "crash open", "kapanıyor": "closes crash", "logoda": "logo splash",
    "arabam": "my car", "arabalarım": "my cars", "kayboldu": "disappeared gone", "silindi": "deleted gone",
    "güncellemeden": "update", "güncelleme": "update", "güncellenmiyor": "not updating",
    "sıralama": "leaderboard", "liderlik": "leaderboard", "tablosu": "leaderboard", "skorum": "score",
    "satın": "bought purchase", "para": "money charged", "coinler": "coins", "gelmedi": "not received",
    "ödedim": "paid", "öğreticide": "tutorial", "kaldım": "stuck", "bitmiyor": "stuck",
    "donuyor": "freezes", "yarış": "race", "alta": "background", "aboneliğim": "vip subscription",
    "bağlantı": "connection", "kopuyor": "disconnects", "çok oyunculu": "multiplayer online",
    "reklam": "ad", "ses": "sound audio", "gelmiyor": "no", "türkçe": "turkish", "garaj": "garage",
    "taşıyor": "text overflow", "sığmıyor": "text overflow button", "günlük": "daily", "ödül": "reward",
    "sayacı": "timer", "sıfırlanıyor": "resets", "uygulama": "app", "oyun": "game",
}

PLATFORM_RE = [
    (re.compile(r"\b(android|samsung|galaxy|pixel|xiaomi|redmi|oppo|google play)\b", re.I), "Android"),
    (re.compile(r"\b(ios|iphone|ipad|ipados|apple)\b", re.I), "iOS"),
]
OS_RE = re.compile(r"\b(android\s?\d+|i(?:pad)?os\s?\d+(?:\.\d+)?)\b", re.I)
GAME_VERSION_RE = re.compile(r"\b(?:v|version\s?)(\d+\.\d+(?:\.\d+)?)\b|\bto (\d+\.\d+\.\d+)\b", re.I)
DEVICE_RE = re.compile(
    r"\b((?:samsung\s)?(?:galaxy\s)?(?:s|a)\d{2}|pixel\s?\d+|xiaomi\s[\w ]*?\d+\w*|redmi note \d+|"
    r"oppo\s\w+|iphone\s?\d+(?:\s?pro)?|ipad[\w ]*?gen|samsung galaxy a\d+)\b", re.I)
SCREENSHOT = ["screenshot", "ss attached", "attached", "receipt", "ekran görüntüsü"]
TURKISH_CHARS = set("çğışöüÇĞİŞÖÜ")
TURKISH_WORDS = {"oyun", "para", "yok", "ama", "lütfen", "sonra", "hata", "bozuk", "var"}


def has_any(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


def is_turkish(text: str) -> bool:
    return any(c in TURKISH_CHARS for c in text) or bool(set(text.lower().split()) & TURKISH_WORDS)


# Game-area concepts: synonyms collapse to one token so short, differently-worded duplicates still match.
CONCEPTS = {
    "zcrash": ["crash", "closes", "closing", "won't start", "wont start", "open the game", "splash", "logo",
               "startup", "açılmıyor", "çöküyor", "kapanıyor", "black screen", "tap the icon"],
    "zcoins": ["coin", "bundle", "iap", "purchase", "charged", "paid", "bought", "refund", "order id"],
    "zvip": ["vip", "subscription", "abonelik", "restore purchase"],
    "zleaderboard": ["leaderboard", "ranking", "record", "best lap", "top 10", "sıralama", "liderlik"],
    "zgarage_car": ["my car", "cars", "garage", "viper", "nova", "araba"],
    "zupdate": ["update", "new version", "patch", "güncelleme", "2.4.0"],
    "ztutorial": ["tutorial", "öğretici", "drift lesson", "hold to drift"],
    "zfreeze": ["freez", "frozen", "donuyor", "background", "switching apps", "phone call", "a call",
                "notifications", "alta alıp"],
    "zonline": ["multiplayer", "online", "pvp", "connection", "disconnect", "bağlantı", "çok oyunculu",
                "3-2-1", "countdown", "start line"],
    "zaudio": ["sound", "audio", "silent", "ses ", "music"],
    "zad": [" ad ", " ad,", "an ad", "the ad", "rewarded video", "reklam"],
    "zdaily": ["daily", "günlük", "timer", "sayacı", "23:59"],
    "zpaint": ["paint", "flicker", "texture", "chrome", "metallic"],
    "zlocale": ["turkish", "türkçe", "yazı", "text goes", "button", "butona", "yükselt"],
}


def concepts(text: str) -> list[str]:
    t = f" {text.lower()} "
    return [c for c, kws in CONCEPTS.items() if any(k in t for k in kws)]


def normalize(text: str) -> str:
    t = text.lower()
    for tr, en in TR_EN.items():
        t = t.replace(tr, f"{tr} {en}")
    return t + (" " + " ".join(c for c in concepts(text) for _ in range(3)) if concepts(text) else "")
