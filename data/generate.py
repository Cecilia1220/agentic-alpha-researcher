"""Generate the deterministic synthetic CSV beside this script."""
import csv
import random
from datetime import date, timedelta
from pathlib import Path


def generate(path):
    rng = random.Random(2421)
    sessions = []
    cursor = date(2024, 1, 2)
    while len(sessions) < 99:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["date", "symbol", "open", "ret_1d", "ret_5d", "volume_shock", "value_score", "value_available"])
        rows = []
        for i in range(8):
            open_price = 65 + 7 * i
            closes = []
            moves = []
            for t, session in enumerate(sessions[:96]):
                # Deliberately weak reversal in the first regime, absent in the second.
                reversal = 0.16 if t < 57 else 0.0
                lagged_move = moves[-2] if len(moves) >= 2 else 0
                open_price *= 1 + rng.gauss(0, 0.005) - reversal * lagged_move
                close = open_price * (1 + rng.gauss(0, 0.009))
                ret_1d = close / closes[-1] - 1 if closes else None
                ret_5d = close / closes[-5] - 1 if len(closes) >= 5 else None
                closes.append(close)
                moves.append(ret_1d or 0)
                volume = rng.gauss(0, 1)
                value = rng.gauss(0, 1)
                release = sessions[t + 2]
                row = [session.isoformat(), f"SYN{i + 1:02d}"]
                row += [f"{x:.7f}" if x is not None else "" for x in (open_price, ret_1d, ret_5d, volume, value)]
                row += [release.isoformat()]
                rows.append(row)
        writer.writerows(sorted(rows))


if __name__ == "__main__":
    generate(Path(__file__).with_name("synthetic_panel.csv"))
