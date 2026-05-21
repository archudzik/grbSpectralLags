"""
Fermi Data Downloader
Downloads TTE data files matching pattern glg_tte_n9_***.fit from
https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/
"""

import os
import re
import sys
import time
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

from config import FERMI_HEASARC_BASE_URL, FERMI_RAW_DIR


class FermiGBMDownloader:
    def __init__(self, base_url=FERMI_HEASARC_BASE_URL):
        self.base_url = base_url
        self.session_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.tte_patterns = [
            re.compile(r'glg_tte_n9_.*\.fit$'),
            re.compile(r'glg_healpix_all_.*\.fit$')
        ]

    def get_page_content(self, url, retries=3):
        """Fetch webpage content with retries"""
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=self.session_headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    return response.read().decode('utf-8')
            except (HTTPError, URLError) as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise

    def extract_links(self, html_content, patterns=None):
        """Extract links from HTML content using a list of patterns"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()

        for link in soup.find_all('a', href=True):
            href = link['href']

            if href and not href.startswith('?') and href != '../':

                if patterns is None:
                    links.add(href)
                else:
                    for pattern in patterns:
                        if pattern.match(href):
                            links.add(href)
                            break

        return list(links)

    def get_year_directories(self):
        """Get list of year directories"""
        print("Fetching year directories...")
        content = self.get_page_content(self.base_url)

        year_pattern = re.compile(r'^\d{4}/$')
        year_dirs = self.extract_links(content, [year_pattern])

        return sorted([year.rstrip('/') for year in year_dirs])

    def get_burst_directories(self, year):
        """Get list of burst directories for a given year"""
        year_url = urllib.parse.urljoin(self.base_url, f"{year}/")
        content = self.get_page_content(year_url)

        burst_pattern = re.compile(r'^bn\d+/$')
        burst_dirs = self.extract_links(content, [burst_pattern])

        return [burst.rstrip('/') for burst in burst_dirs]

    def get_tte_files(self, year, burst):
        """Get list of TTE files for a specific burst"""
        current_url = urllib.parse.urljoin(
            self.base_url, f"{year}/{burst}/current/")

        try:
            content = self.get_page_content(current_url)
            tte_files = self.extract_links(content, self.tte_patterns)
            return tte_files
        except Exception as e:
            print(f"Could not access {current_url}: {e}")
            return []

    def download_file(self, url, local_path, chunk_size=8192):
        """Download file with progress indication"""
        try:
            req = urllib.request.Request(url, headers=self.session_headers)

            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                with open(local_path, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(
                                f"\r  Downloading: {percent:.1f}%", end='', flush=True)

                print(f"\r  Downloaded: {os.path.basename(local_path)}")
                return True

        except Exception as e:
            print(f"\r  Failed to download {url}: {e}")
            return False

    def run(self, download_dir=FERMI_RAW_DIR, start_year=None, end_year=None):
        """Main execution function"""
        print("Fermi GBM TTE Data Downloader")
        print(f"Target directory: {download_dir}")

        years = self.get_year_directories()

        if start_year:
            years = [y for y in years if int(y) >= start_year]
        if end_year:
            years = [y for y in years if int(y) <= end_year]

        print(f"Processing years: {years}")

        total_files = 0
        successful_downloads = 0

        for year in years:
            print(f"\nProcessing year {year}...")

            try:
                burst_dirs = self.get_burst_directories(year)
                print(f"Found {len(burst_dirs)} burst directories in {year}")

                for i, burst in enumerate(burst_dirs):
                    print(f"  [{i+1}/{len(burst_dirs)}] Checking {burst}...")

                    tte_files = self.get_tte_files(year, burst)

                    if tte_files:
                        print(f"    Found {len(tte_files)} TTE files")

                        for tte_file in tte_files:
                            file_url = urllib.parse.urljoin(
                                self.base_url,
                                f"{year}/{burst}/current/{tte_file}"
                            )

                            local_path = os.path.join(
                                download_dir,
                                year,
                                burst,
                                tte_file
                            )

                            if os.path.exists(local_path):
                                print(
                                    f"    Skipping {tte_file} (already exists)")
                                continue

                            total_files += 1
                            if self.download_file(file_url, local_path):
                                successful_downloads += 1

                            time.sleep(0.5)

            except KeyboardInterrupt:
                print("\nDownload interrupted by user")
                break
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue

        print(f"\nDownload complete!")
        print(f"Total files processed: {total_files}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {total_files - successful_downloads}")


def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Fermi GBM TTE data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fermi_download.py
  python fermi_download.py --start-year 2020
  python fermi_download.py --start-year 2020 --end-year 2022
  python fermi_download.py --download-dir ./data
        """
    )

    parser.add_argument(
        '--download-dir',
        default=FERMI_RAW_DIR,
        help=f'Directory to save downloaded files (default: {FERMI_RAW_DIR})'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Start year (inclusive)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='End year (inclusive)'
    )

    args = parser.parse_args()

    downloader = FermiGBMDownloader()

    try:
        downloader.run(
            download_dir=args.download_dir,
            start_year=args.start_year,
            end_year=args.end_year
        )
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
