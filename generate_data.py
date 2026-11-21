"""Generate a synthetic Discord bug-report dataset for the fictional game "Neon Drift".

Everything here is invented. No real player messages, IDs or usernames.

Outputs:
  data/messages.json      raw messages, as the pipeline would receive them
  data/ground_truth.json  labels: message id -> type, bug_id, severity
  data/planted_bugs.json  the 12 underlying bugs with their true severity
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent / "data"

ANDROID = [("Samsung Galaxy A52", "Android 13"), ("Xiaomi Redmi Note 12", "Android 13"),
           ("Pixel 7", "Android 14"), ("Samsung S23", "Android 14"), ("Oppo A78", "Android 13"),
           ("Galaxy A34", "Android 14")]
ANDROID14 = [("Pixel 7", "Android 14"), ("Samsung S23", "Android 14"), ("Galaxy A54", "Android 14"),
             ("Xiaomi 13T", "Android 14")]
IOS = [("iPhone 13", "iOS 17.5"), ("iPhone 11", "iOS 17.4"), ("iPhone 15 Pro", "iOS 18.1"),
       ("iPad 9th gen", "iPadOS 17.5")]
ANY = ANDROID + IOS
VERSIONS = ["2.4.0", "2.4.1"]

# id, title, true severity, device pool, report texts (each one is a distinct player message)
PLANTED_BUGS = [
    ("B01", "Coin pack purchase charged but coins not delivered", "S1", ANY, [
        "bought the 500 coin pack, got charged but no coins showed up??",
        "I paid 4.99 for the coin bundle and my balance didn't change. money gone",
        "purchase went through on google play but 0 coins in game. screenshot of receipt attached",
        "satın aldım para çekildi ama coinler gelmedi",
        "Charged twice for mega coin pack and recieved nothing. need refund asap",
        "coin pack purchase stuck on 'processing' then nothing, but bank says charged",
        "where are my coins i literally paid for them",
        "Bought starter bundle - payment confirmed email from apple, no coins, no car",
        "para ödedim coin yok!! dolandırıcılık mı bu",
        "IAP bug: 1200 coins pack charged, balance still 340. ss attached",
        "spent 9.99 on coins and got nothing, restarting doesnt help",
        "after buying coins the game froze and when i came back coins were not added but i was charged",
        "my son bought coins with my card, charge shows on statement, nothing in the game",
        "coins purchase not delivered. order id GPA.0000-1111-2222 (fake example)",
    ]),
    ("B02", "Crash on launch on Android 14", "S1", ANDROID14, [
        "game crashes on startup since my phone updated to android 14",
        "opens then instantly closes. Pixel 7",
        "crash at the loading logo every time. samsung s23 android 14",
        "oyun açılmıyor, logoda kapanıyor. android 14",
        "cant even open the game anymore, black screen then back to home screen",
        "Crash on launch after Android 14 update, reinstalled twice, same",
        "app keeps closing on splash screen pls fix",
        "uygulama açılır açılmaz çöküyor",
        "game crashes immediatly when i open it",
        "Launch crash - Galaxy A54, Android 14, v2.4.1. Worked fine on Android 13",
        "it just closes itself when i tap the icon. tried clearing cache",
        "won't start, crashes before main menu",
    ]),
    ("B03", "Leaderboard not updating after races", "S2", ANY, [
        "leaderboard isnt updating, my time from yesterday still not there",
        "weekly leaderboard stuck, same top 10 for 3 days",
        "set a new record on Neon Bridge but leaderboard shows my old time",
        "sıralama tablosu güncellenmiyor",
        "leaderboard frozen?? nobody moved since monday",
        "my best lap doesnt save to the leaderboard",
        "Leaderboard not refreshing after races, tried logging out and in",
        "global ranking hasn't changed in days",
        "liderlik tablosunda skorum görünmüyor",
    ]),
    ("B04", "Owned car missing from garage after 2.4.0 update", "S1", ANY, [
        "after the update my Viper GT is gone from the garage!!",
        "updated to 2.4.0 and lost my car I bought last week",
        "güncellemeden sonra arabam kayboldu",
        "my legendary car disappeared after update, i spent months on it",
        "garage is missing 2 cars since the new version",
        "where did my Nova R go?? update deleted it",
        "Car vanished from garage after patch, upgrades too",
        "arabalarım silindi güncelleme sonrası",
        "lost my fully upgraded car after updating. not cool",
        "the update ate my car lol. it's not in the garage anymore",
    ]),
    ("B05", "Tutorial soft-lock at drift lesson", "S2", ANY, [
        "stuck in tutorial at the drift part, it never tells me to continue",
        "new player, tutorial wont let me pass the drift lesson",
        "tutorial drift step: completed the drift but nothing happens, cant leave",
        "öğreticide kaldım, drift kısmı bitmiyor",
        "cant finish tutorial, stuck forever on 'hold to drift'",
        "friend downloaded game and is stuck in tutorial, no skip button",
        "Tutorial softlock - drift lesson doesn't register the drift",
        "stuck on tutorial cant play the game at all",
    ]),
    ("B06", "Race freezes after returning from background", "S2", ANY, [
        "if i get a call during a race the game freezes when i come back",
        "race freezes after switching apps, have to force close",
        "oyunu alta alıp geri açınca yarış donuyor",
        "game freezes when returning from background mid race",
        "pulled down notifications mid race -> frozen screen",
        "freezing after phone call during races, have to kill the app",
    ]),
    ("B07", "VIP subscription not restored on new device / reinstall", "S1", ANY, [
        "got a new phone and my VIP subscription is gone, restore purchases does nothing",
        "VIP pass paid monthly but game says I'm not VIP on my tablet",
        "restore purchase button not working for VIP subscription",
        "VIP aboneliğim yeni telefonda görünmüyor, para ödüyorum",
        "still being charged for VIP but lost VIP after reinstall",
    ]),
    ("B08", "Multiplayer disconnect at race start", "S2", ANY, [
        "multiplayer disconnects right when the countdown ends",
        "keep getting 'connection lost' at the start of online races",
        "online race kicks me at 3-2-1, wifi is fine",
        "çok oyunculu yarışta başlarken bağlantı kopuyor",
        "every online match disconnects at start, lost my win streak",
        "PvP races dropping connection at the start line",
        "disconnected at race start 5 times in a row",
    ]),
    ("B09", "Audio lost after rewarded ad", "S3", ANY, [
        "no sound after watching an ad for double reward",
        "music and engine sounds gone after rewarded video",
        "reklam izledikten sonra ses gelmiyor",
        "audio cuts out after ad until i restart the game",
        "after the ad the game is silent",
    ]),
    ("B10", "Turkish text overflows buttons in garage menu", "S4", ANY, [
        "Türkçe dilinde garaj menüsünde yazılar taşıyor",
        "In turkish language the upgrade button text goes outside the button",
        "garaj ekranında 'Yükselt' yazısı butona sığmıyor",
    ]),
    ("B11", "Daily reward timer resets on app open", "S3", ANY, [
        "daily reward timer resets every time i open the app",
        "can't claim daily reward, timer goes back to 23:59",
        "günlük ödül sayacı sürekli sıfırlanıyor",
        "daily bonus says claimed but i never got it and timer restarted",
    ]),
    ("B12", "Metallic car paint flickers in garage on iOS", "S4", IOS, [
        "chrome paint flickers in the garage on my iphone",
        "car paint texture flashing/flickering on iOS",
        "metallic paint looks glitchy, flickers when rotating the car. iPhone 13",
    ]),
]

VAGUE_BUGS = [
    "game broken pls fix", "doesnt work", "bug!!!", "oyun bozuk düzeltin",
    "something is wrong with the game since yesterday", "pls fix the bugs",
    "it glitched and now idk what to do", "hata var",
]
FEATURES = [
    "please add a photo mode", "can we get more tracks in the city", "add clan/team races",
    "dark mode for menus pls", "would love a manual gear option", "yeni araba ekleyin lütfen",
    "add controller support", "let us trade cars with friends", "a replay feature would be cool",
    "add rain weather to tracks", "suggestion: season reset rewards for ranked",
    "custom license plates!!", "türkçe dublaj olsa güzel olur", "make tournaments weekly please",
    "option to turn off motion blur would be nice",
]
COMPLAINTS = [
    "too many ads honestly", "matchmaking puts me vs people way stronger than me",
    "energy system is so greedy", "the new car is pay to win", "reklamlar çok fazla",
    "loved the new neon track!!", "game is great but the grind is too much",
    "why is the new update so big", "prices in the shop are crazy",
    "controls feel worse since update", "best racing game on mobile tbh",
    "event rewards are trash this week", "bu oyun çok güzel ama çok pahalı",
    "drift physics feel floaty now", "too hard to get legendary cars without paying",
]
QUESTIONS = [
    "how do i unlock the night city track?", "when does the season end?",
    "is there a way to transfer progress to iOS?", "what is the best car for drift events?",
    "nasıl kulüp kurulur?", "do coins carry over to next season?", "how do i change my username",
    "when is the next update coming", "is cross play a thing?", "how to get more fuel",
    "what does the turbo stat do", "can i play offline?",
]
SPAM = [
    "FREE COINS generator click neon-free-coins.example", "selling account lvl 80 dm me",
    "join my discord server!!", "sub to my youtube for drift tips", "aaaaaaaaaaa", "first",
    "🔥🔥🔥", "anyone wanna be friends", "ücretsiz coin için linke tıkla", "lol",
]


def _env_suffix(rng: random.Random, pool) -> str:
    """Players sometimes add device / OS / version info, usually they don't."""
    if rng.random() > 0.45:
        return ""
    device, os_ver = rng.choice(pool)
    ver = rng.choice(VERSIONS)
    style = rng.randrange(5)
    return [
        f" - {device}, {os_ver}, v{ver}",
        f" ({device} / {os_ver})",
        f" version {ver}",
        f" {os_ver}",
        f". phone: {device}",
    ][style]


def generate(seed: int = SEED) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str, str | None, str | None]] = []  # (text, type, bug_id, severity)
    for bug_id, _title, sev, pool, texts in PLANTED_BUGS:
        for t in texts:
            text = t + _env_suffix(rng, pool)
            if rng.random() < 0.1 and "attached" not in text:
                text += " (screenshot attached)"
            rows.append((text, "bug", bug_id, sev))
    rows += [(t, "bug", None, None) for t in VAGUE_BUGS]
    rows += [(t, "feature_request", None, None) for t in FEATURES]
    rows += [(t, "feedback", None, None) for t in COMPLAINTS]
    rows += [(t, "question", None, None) for t in QUESTIONS]
    rows += [(t, "spam", None, None) for t in SPAM]
    rng.shuffle(rows)

    channels = {"bug": "#bug-reports", "feature_request": "#suggestions"}
    start = datetime(2026, 11, 20, 0, 5)
    messages, truth = [], []
    for i, (text, typ, bug_id, sev) in enumerate(rows, start=1):
        mid = f"m{i:03d}"
        ts = start + timedelta(minutes=i * 9 + rng.randrange(8))
        # ~20% of messages land in the "wrong" channel, as on a real server
        channel = channels.get(typ, "#general") if rng.random() > 0.2 else rng.choice(
            ["#general", "#bug-reports", "#suggestions"])
        messages.append({
            "id": mid,
            "author": f"racer_{rng.randrange(1000, 9999)}",
            "channel": channel,
            "timestamp": ts.isoformat(timespec="minutes"),
            "text": text,
        })
        truth.append({"id": mid, "type": typ, "bug_id": bug_id, "severity": sev})
    bugs = [{"bug_id": b, "title": t, "severity": s, "report_count": len(x)}
            for b, t, s, _p, x in PLANTED_BUGS]
    return messages, truth, bugs


def main() -> None:
    messages, truth, bugs = generate()
    DATA_DIR.mkdir(exist_ok=True)
    for name, obj in [("messages.json", messages), ("ground_truth.json", truth),
                      ("planted_bugs.json", bugs)]:
        (DATA_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"Wrote {len(messages)} messages to {DATA_DIR}")
    print("By type:", dict(Counter(t["type"] for t in truth)))
    print("Planted bugs:", ", ".join(f"{b['bug_id']}({b['severity']}) x{b['report_count']}" for b in bugs))


if __name__ == "__main__":
    main()
