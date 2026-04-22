from pathlib import Path
import os
import time

import pandas as pd
import requests

API_KEY = os.environ.get("KNMI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        'KNMI_API_KEY is not set. In WSL, run: export KNMI_API_KEY="your_key_here"'
    )

DATASET = "10-minute-in-situ-meteorological-observations"
VERSION = "1.0"
DEST = Path("data_raw/knmi_10min")
STATE_FILE = Path("data_raw/knmi_10min_last.txt")

START_UTC = pd.Timestamp("2012-05-01 00:00:00", tz="UTC")
END_UTC = pd.Timestamp("2012-09-30 23:50:00", tz="UTC")

MAX_RETRIES = 8

# This delay applies to KNMI API calls to the /url endpoint.
# 5 s gives about 720 KNMI API calls/hour, below the 1000/hour limit.
MIN_SECONDS_BETWEEN_API_CALLS = 5.0

BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"
_last_api_call_time = 0.0


def throttle():
    global _last_api_call_time
    now = time.time()
    dt = now - _last_api_call_time
    if dt < MIN_SECONDS_BETWEEN_API_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_API_CALLS - dt)
    _last_api_call_time = time.time()


def load_state():
    if STATE_FILE.exists():
        text = STATE_FILE.read_text().strip()
        if text:
            return text
    return None


def save_state(last_downloaded_filename):
    STATE_FILE.write_text("" if last_downloaded_filename is None else last_downloaded_filename)


def filename_from_timestamp(ts):
    return f"KMDS__OPER_P___10M_OBS_L2_{ts.strftime('%Y%m%d%H%M')}.nc"


def backoff_delay(response, attempt):
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return min(60.0 * (2 ** attempt), 1800.0)


def iter_timestamps_in_range(start_utc, end_utc):
    if end_utc < start_utc:
        raise ValueError("END_UTC must be >= START_UTC")

    current = start_utc
    step = pd.Timedelta(minutes=10)

    while current <= end_utc:
        yield current
        current += step


def get_temporary_download_url(session, filename):
    url = (
        f"{BASE_URL}/datasets/{DATASET}/versions/{VERSION}/files/{filename}/url"
    )

    for attempt in range(MAX_RETRIES):
        throttle()

        response = session.get(url, timeout=60)

        if response.status_code == 200:
            data = response.json()
            return data["temporaryDownloadUrl"]

        if response.status_code == 404:
            return None

        if response.status_code == 429 and attempt < MAX_RETRIES - 1:
            delay = backoff_delay(response, attempt)
            print(f"429 while requesting URL for {filename}. Retrying in {delay:.1f} s ...")
            time.sleep(delay)
            continue

        response.raise_for_status()

    raise RuntimeError(f"Failed to obtain temporary URL for {filename}")


def download_from_temporary_url(temp_url, outpath):
    with requests.get(temp_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_one_with_backoff(session, filename, outpath):
    temp_url = get_temporary_download_url(session, filename)
    if temp_url is None:
        return False

    download_from_temporary_url(temp_url, outpath)
    return True


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Authorization": API_KEY})

    resume_after = load_state()

    if resume_after:
        print(f"Resuming after: {resume_after}")
    else:
        print(f"Starting fresh: {START_UTC} to {END_UTC}")

    downloaded = 0
    skipped_existing = 0
    skipped_missing = 0
    total_seen = 0
    resuming = resume_after is not None

    for ts in iter_timestamps_in_range(START_UTC, END_UTC):
        filename = filename_from_timestamp(ts)

        if resuming:
            if filename <= resume_after:
                continue
            resuming = False

        total_seen += 1
        outpath = DEST / filename

        if outpath.exists():
            skipped_existing += 1
            save_state(filename)
            continue

        ok = download_one_with_backoff(session, filename, outpath)

        if ok:
            downloaded += 1
            save_state(filename)

            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded} new files; last={filename}")
        else:
            skipped_missing += 1
            save_state(filename)

    print("\nDone.")
    print("Seen in this run:", total_seen)
    print("Downloaded new files:", downloaded)
    print("Skipped existing files:", skipped_existing)
    print("Skipped missing files:", skipped_missing)
    print("Folder:", DEST.resolve())


if __name__ == "__main__":
    main()