import os
from pathlib import Path

try:
    import requests
except ImportError:
    raise ImportError("The requests library is required. Install it with: pip install requests")

DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESOURCES = {
    "2024_BRFSS_main_documentation.html": "https://www.cdc.gov/brfss/annual_data/annual_2024.html",
    "2024_BRFSS_overview.pdf": "https://www.cdc.gov/brfss/annual_data/2024/pdf/Overview_2024-508.pdf",
    "2024_variable_layout.html": "https://www.cdc.gov/brfss/annual_data/2024/llcp_varlayout_24_onecolumn_v2.html",
    "2024_calculated_variables.pdf": "https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf",
    "BRFSS_questionnaires_hub.html": "https://www.cdc.gov/brfss/questionnaires/index.htm",
    "2024_modules_by_category.html": "https://www.cdc.gov/brfss/questionnaires/modules/category2024.htm",
    "2024_modules_by_state.html": "https://www.cdc.gov/brfss/questionnaires/modules/state2024.htm",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BRFSS-Downloader/1.0; +https://example.com)"
}


def download_resource(filename: str, url: str) -> None:
    destination = DOWNLOAD_DIR / filename
    response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Saved: {destination}")


if __name__ == "__main__":
    for filename, url in RESOURCES.items():
        try:
            download_resource(filename, url)
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")
