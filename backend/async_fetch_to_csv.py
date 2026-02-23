"""
Async fetch‑to‑CSV utility.

Provides an asynchronous function to fetch JSON data from a list of URLs
and write the combined results to a CSV file.
"""

import asyncio
import csv
import aiohttp
from typing import List, Dict


async def fetch(session: aiohttp.ClientSession, url: str) -> Dict:
    """Fetch JSON data from a single URL."""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()


async def fetch_all(urls: List[str]) -> List[Dict]:
    """Fetch data from all URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


def write_to_csv(data: List[Dict], csv_path: str) -> None:
    """Write a list of dictionaries to a CSV file."""
    if not data:
        return
    # Use the keys from the first record as the CSV header.
    fieldnames = data[0].keys()
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main(urls: List[str], csv_path: str) -> None:
    """Entry point: fetch URLs and store results in CSV."""
    results = asyncio.run(fetch_all(urls))
    write_to_csv(results, csv_path)


if __name__ == "__main__":
    # Placeholder – replace with real URLs and output path as needed.
    sample_urls = []
    output_csv = "output.csv"
    main(sample_urls, output_csv)
