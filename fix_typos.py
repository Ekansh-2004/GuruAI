import json

PROFILE_FILE = "knowledge_profile.json"
with open(PROFILE_FILE, "r") as f:
    data = json.load(f)

for subj, topics in data.items():
    if "Data Fudiciary" in topics:
        stats = topics.pop("Data Fudiciary")
        if "Data Fiduciary" in topics:
            topics["Data Fiduciary"]["correct"] += stats["correct"]
            topics["Data Fiduciary"]["total"] += stats["total"]
            # average ema_score roughly
            t_ema = topics["Data Fiduciary"]["ema_score"]
            s_ema = stats["ema_score"]
            if t_ema is not None and s_ema is not None:
                topics["Data Fiduciary"]["ema_score"] = (t_ema + s_ema) / 2
            elif s_ema is not None:
                topics["Data Fiduciary"]["ema_score"] = s_ema
        else:
            topics["Data Fiduciary"] = stats

with open(PROFILE_FILE, "w") as f:
    json.dump(data, f, indent=2)
print("Fixed knowledge_profile.json")
