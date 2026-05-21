"""
Swift BAT Data Downloader
Downloads Swift/BAT event files used for spectral-lag analysis.
https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/
"""

import os
import re
import sys
import time
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

from config import SWIFT_HEASARC_BASE_URL, SWIFT_RAW_DIR

class SwiftBATDownloader:
    def __init__(self):
        self.base_url = SWIFT_HEASARC_BASE_URL
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.bat_subdirs = ['event']
        self.file_patterns = [
            re.compile(r'.*bevshsp_uf\.evt(\.gz)?$')
        ]

    @staticmethod
    def normalize_year_month(value):
        if value is None:
            return None

        value = str(value).strip()
        patterns = [
            re.compile(r'^(\d{4})_(\d{2})$'),
            re.compile(r'^(\d{4})-(\d{2})(?:-\d{2})?$'),
            re.compile(r'^(\d{4})/(\d{2})(?:/\d{2})?$'),
        ]
        for pattern in patterns:
            match = pattern.match(value)
            if match:
                year, month = match.groups()
                month_int = int(month)
                if not 1 <= month_int <= 12:
                    raise ValueError(f"Invalid month in date: {value}")
                return f"{year}_{month_int:02d}"

        raise ValueError(
            f"Invalid date format: {value}. Use YYYY_MM or YYYY-MM-DD."
        )

    def get_page(self, url, silent_404=True, retries=4, backoff=2.0):
        """Fetch page content"""
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=120) as response:
                    return response.read().decode('utf-8')
            except HTTPError as e:
                if e.code == 404 and silent_404:
                    return None
                last_error = e
            except (URLError, ConnectionResetError, TimeoutError) as e:
                last_error = e
            except Exception as e:
                last_error = e

            if attempt < retries:
                wait_time = backoff * attempt
                print(
                    f"      retry {attempt}/{retries - 1} after error: "
                    f"{last_error}"
                )
                time.sleep(wait_time)

        if silent_404:
            print(f"      failed to fetch {url}: {last_error}")
            return None
        raise last_error

    def get_links(self, html):
        """Extract file links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        files = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href and not href.startswith('?') and href not in ['../', '/']:
                if not href.endswith('/'):
                    files.append(href)
        return files

    def get_year_months(self):
        """Get all year_month directories"""
        content = self.get_page(self.base_url, silent_404=False)
        soup = BeautifulSoup(content, 'html.parser')

        ym_pattern = re.compile(r'^\d{4}_\d{2}/$')
        year_months = []

        for link in soup.find_all('a', href=True):
            if ym_pattern.match(link['href']):
                year_months.append(link['href'].rstrip('/'))

        return sorted(year_months)

    def get_observations(self, year_month):
        """Get observations for a year_month"""
        url = urllib.parse.urljoin(self.base_url, f"{year_month}/")
        content = self.get_page(url, silent_404=False)

        soup = BeautifulSoup(content, 'html.parser')
        obs_pattern = re.compile(r'^\d{11}/$')
        observations = []

        for link in soup.find_all('a', href=True):
            if obs_pattern.match(link['href']):
                observations.append(link['href'].rstrip('/'))

        return observations

    def get_bat_files(self, year_month, obs_id, subdir):
        """Get BAT files from a specific subdirectory"""
        url = urllib.parse.urljoin(
            self.base_url,
            f"{year_month}/{obs_id}/bat/{subdir}/"
        )

        content = self.get_page(url, silent_404=True)
        if content is None:
            return []

        all_files = self.get_links(content)

        if self.file_patterns:
            filtered = []
            for f in all_files:
                for pattern in self.file_patterns:
                    if pattern.match(f):
                        filtered.append(f)
                        break
            return filtered

        return all_files

    def download_file(self, url, local_path):
        """Download a file"""
        try:
            req = urllib.request.Request(url, headers=self.headers)

            with urllib.request.urlopen(req, timeout=120) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                with open(local_path, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(
                                f"\r      Progress: {pct:.1f}%", end='', flush=True)

                print(f"\r      OK {os.path.basename(local_path)}")
                return True

        except Exception as e:
            print(f"\r      FAIL Failed: {e}")
            return False

    def run(self, download_dir=SWIFT_RAW_DIR, start_date=None, end_date=None,
            grb_only=False, subdirs=None):
        """
        Main download function

        Args:
            download_dir: Output directory
            start_date: Start YYYY_MM or YYYY-MM-DD
            end_date: End YYYY_MM or YYYY-MM-DD
            grb_only: Only obs ending in 000
            subdirs: List of subdirs to download (default: all)
        """
        print("Swift BAT Data Downloader")
        print(f"Output: {download_dir}")

        if subdirs is None:
            subdirs = self.bat_subdirs
        print(f"Subdirectories: {', '.join(subdirs)}")
        start_date = self.normalize_year_month(start_date)
        end_date = self.normalize_year_month(end_date)

        year_months = self.get_year_months()
        if start_date:
            year_months = [ym for ym in year_months if ym >= start_date]
        if end_date:
            year_months = [ym for ym in year_months if ym <= end_date]

        print(f"Date range: {year_months[0]} to {year_months[-1]}")
        print(f"Months: {len(year_months)}\n")

        stats = {
            'total_obs': 0,
            'obs_with_data': 0,
            'files_downloaded': 0,
            'files_skipped': 0,
            'files_failed': 0
        }

        for ym in year_months:
            print(f"\n{'='*70}")
            print(f"Processing {ym}")
            print('='*70)

            try:
                observations = self.get_observations(ym)

                if grb_only:
                    observations = [
                        o for o in observations if o.endswith('000')]

                print(f"Observations: {len(observations)}")

                for i, obs_id in enumerate(observations):
                    stats['total_obs'] += 1

                    if (i + 1) % 25 == 0:
                        print(f"  Progress: {i+1}/{len(observations)}")

                    obs_has_data = False

                    for subdir in subdirs:
                        try:
                            files = self.get_bat_files(ym, obs_id, subdir)
                        except Exception as e:
                            print(
                                f"  {obs_id}: failed to list {subdir}/ "
                                f"after retries: {e}"
                            )
                            stats['files_failed'] += 1
                            continue

                        if files:
                            if not obs_has_data:
                                print(f"\n  {obs_id}:")
                                obs_has_data = True

                            print(f"    {subdir}/: {len(files)} files")

                            for filename in files:
                                file_url = urllib.parse.urljoin(
                                    self.base_url,
                                    f"{ym}/{obs_id}/bat/{subdir}/{filename}"
                                )

                                local_path = os.path.join(
                                    download_dir, ym, obs_id, 'bat', subdir, filename
                                )

                                if os.path.exists(local_path):
                                    print(f"      SKIP {filename} (exists)")
                                    stats['files_skipped'] += 1
                                    continue

                                if self.download_file(file_url, local_path):
                                    stats['files_downloaded'] += 1
                                else:
                                    stats['files_failed'] += 1

                                time.sleep(0.3)

                    if obs_has_data:
                        stats['obs_with_data'] += 1

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"Observations checked:     {stats['total_obs']}")
        print(f"Observations with data:   {stats['obs_with_data']}")
        print(f"Files downloaded:         {stats['files_downloaded']}")
        print(f"Files skipped (existing): {stats['files_skipped']}")
        print(f"Files failed:             {stats['files_failed']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Swift BAT data",
        epilog="""
Examples:
  python swift_download.py --start-date 2024_01
  
  python swift_download.py --start-date 2024-01-01 --end-date 2024-12-31
  
  python swift_download.py --download-dir ./swift_bat_data
        """
    )

    parser.add_argument('--download-dir', default=SWIFT_RAW_DIR,
                        help='Output directory')
    parser.add_argument('--start-date', help='Start YYYY_MM or YYYY-MM-DD')
    parser.add_argument('--end-date', help='End YYYY_MM or YYYY-MM-DD')
    parser.add_argument('--grb-only', action='store_true',
                        help='Only obs ending in 000')
    parser.add_argument('--subdirs', nargs='+',
                        choices=['event', 'rate', 'survey', 'masktag', 'hk'],
                        help='BAT subdirectories to download (default: event)')

    args = parser.parse_args()

    downloader = SwiftBATDownloader()

    try:
        downloader.run(
            download_dir=args.download_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            grb_only=args.grb_only,
            subdirs=args.subdirs
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
