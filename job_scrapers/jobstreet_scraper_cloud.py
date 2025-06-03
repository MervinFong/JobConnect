import cloudscraper
from bs4 import BeautifulSoup
import sys
import json

def scrape_jobstreet_jobs_cloud(keyword: str, location: str = "Malaysia", max_pages: int = 1):
    results = []
    keyword_slug = keyword.lower().replace(" ", "-")
    location_slug = location.lower().replace(" ", "-")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    scraper = cloudscraper.create_scraper()

    for page in range(max_pages):
        start = page * 20
        url = f"https://www.jobstreet.com.my/en/job-search/{keyword_slug}-jobs/in-{location_slug}?start={start}"
        print(f"[DEBUG] Fetching: {url}", file=sys.stderr, flush=True)

        try:
            response = scraper.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[DEBUG] Failed to fetch: {response.status_code}", file=sys.stderr, flush=True)
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            job_cards = soup.select("a[data-automation='job-list-view-job-link']")
            print(f"[DEBUG] Found {len(job_cards)} job cards", file=sys.stderr, flush=True)

            for job in job_cards:
                href = job.get("href", "")
                if not href:
                    continue

                full_link = href if href.startswith("http") else "https://www.jobstreet.com.my" + href
                results.append({"link": full_link})

        except Exception as e:
            print(f"❌ Error during scraping: {e}", file=sys.stderr, flush=True)
            continue

    return results

# ✅ Entry point for subprocess execution
if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else "Software Engineer"
    location = sys.argv[2] if len(sys.argv) > 2 else "Malaysia"

    scraped = scrape_jobstreet_jobs_cloud(keyword, location)
    sys.stdout.write(json.dumps(scraped))
    sys.stdout.flush()
