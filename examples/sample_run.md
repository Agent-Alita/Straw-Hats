# Sample Run

```bash
uv run straw-hats \
  --poem ./examples/poem.txt \
  --reddit "https://www.reddit.com/r/sanfrancisco/comments/XXXXXX/treasure_hunt_poem/" \
  --out ./out/report.md \
  --json ./out/answer.json \
  --max-turns 30
```

Expected stdout (abridged):

```
╭────────────────────────────────────╮
│ straw-hats hunting in San Francisco │
│ Reddit: https://www.reddit.com/... │
│ Poem chars: 412                    │
╰────────────────────────────────────╯
→ tool reddit_thread {"url": "https://www.reddit.com/r/..."}
← reddit_thread {"op": {"title": "...", "score": 412, ...}, "comments": [...]}
→ tool wikipedia_lookup {"query": "Buena Vista Park San Francisco"}
← wikipedia_lookup {"pages": [...]}
→ tool geocode {"query": "Buena Vista Park east overlook"}
← geocode {"results": [{"display_name": "...", "lat": 37.7686, "lng": -122.4416}]}
→ tool nearby_search {"lat": 37.7686, "lng": -122.4416, "query": "plaque", "radius_m": 200}
...
assistant: ```json
{
  "location_name": "Buena Vista Park — east overlook",
  "lat": 37.7686, "lng": -122.4416,
  "confidence": 0.72,
  ...
}
```
## Reasoning
...
```

The markdown report and JSON file land in `./out/`.
