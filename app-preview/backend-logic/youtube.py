from typing import List, Dict, Any, Optional
import httpx

YOUTUBE_COMMENTS_API = "https://www.googleapis.com/youtube/v3/commentThreads"

async def fetch_youtube_comments(
    api_key: str,
    video_id: str,
    max_results: int = 50,
    page_token: Optional[str] = None
) -> Dict[str, Any]:
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max(1, min(max_results, 100)),
        "textFormat": "plainText",
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(YOUTUBE_COMMENTS_API, params=params)
        r.raise_for_status()
        data = r.json()

    comments: List[Dict[str, Any]] = []
    for item in data.get("items", []):
        try:
            snip = item["snippet"]["topLevelComment"]["snippet"]
        except (KeyError, TypeError):
            continue
        comments.append({
            "comment_id": item["snippet"]["topLevelComment"]["id"],
            "author": snip.get("authorDisplayName"),
            "text": snip.get("textDisplay", ""),
            "like_count": snip.get("likeCount", 0),
            "published_at": snip.get("publishedAt"),
        })

    return {
        "comments": comments,
        "next_page_token": data.get("nextPageToken"),
        "total_results": data.get("pageInfo", {}).get("totalResults"),
    }
