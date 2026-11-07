# Addon mode (one key, sound only)

Fishing addons such as [Better Fishing](https://www.curseforge.com/wow/addons/better-fishing)
let one key do both jobs: press it to cast, and press it again while the bobber is out to reel
in. With such an addon the bot does not need to find the bobber on screen at all, so addon mode
skips the gear icon search and only listens for the splash.

## How it works

1. **Cast**: the bot presses `cast_key`.
2. **Listen**: it waits for the splash, exactly like the normal mode (same detector, same
   `bite_threshold`). No screenshots are taken and no calibration is needed.
3. **Reel in**: when it hears the splash it presses `cast_key` again. The addon turns that
   press into an interact with the bobber, and Auto Loot picks up the fish.
4. **Loot**: the bot gives Auto Loot one second, counts a catch, and starts over.

If no splash is heard within `cast_timeout_s`, the cast counts as a miss and the bot casts again.

## Setup

1. Install a one-key fishing addon (for example Better Fishing) and bind its key, or the
   Fishing ability it uses, to `cast_key`. Check by hand in game: one press casts, a second
   press while the bobber is out reels in.
2. Turn on Auto Loot (Options > Controls > Auto Loot).
3. Turn addon mode on in the config file (`wowfish config --path` prints where it is):

   ```toml
   [game]
   cast_key = "1"
   addon_mode = true
   ```

   `interact_key` and everything under `[vision]` are not used in this mode.

## Other options

Everything under `[run]` works the same as in the normal mode:

| Option | Effect in addon mode |
|---|---|
| `pause_min_s`, `pause_max_s` | random pause before each cast |
| `break_chance`, `break_min_s`, `break_max_s` | occasional longer break |
| `reaction_min_s`, `reaction_max_s` | random wait between the splash and the second press |
| `skip_bite_chance` | let some bites go on purpose (counted as misses); the bot waits 3 s for the fish to get away before it casts again, so the next press casts instead of reeling in |
| `key_hold_ms`, `key_hold_jitter_ms` | how long each key is held |
| `move_chance`, `move_keys`, `move_max_ms` | small step and back after a catch |
| `cast_timeout_s` | how long to wait for a splash |
| `session_minutes` (under `[game]`) | stop after this many minutes |

`icon_timeout_s` and `loot_timeout_s` are not used: there is no icon to find or to watch.

## Limits

- The bot cannot see whether the fish was really looted; every splash it reacts to counts as
  a catch.
- Splashes from other players' bobbers nearby can trigger a press, because there is no bobber
  position to check against. Fish away from other anglers.
