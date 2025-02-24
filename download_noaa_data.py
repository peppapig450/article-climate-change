from __future__ import annotations

import asyncio
import gzip
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

import aiofiles
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

# Type aliasees
type FilePath = Path | str
type Year = int
type URL = str


class DirectoryType(StrEnum):
    """Enumeration of directory types used in the script."""

    RAW = auto()
    EXTRACTED = auto()
    COMBINED = auto()


@dataclass(frozen=True)
class FileMetadata:
    """Data class for storing file metadata."""

    filename: str
    year: int
    url: str
    output_path: Path

    @classmethod
    def from_filename(
        cls, filename: str, base_url: str, output_dir: Path
    ) -> FileMetadata:
        """Create a FileMetadata instance from a filename.

        Args:
            filename: The filename from which to extract metadata.
            base_url: The base URL to construct the file URL.
            output_dir: The directory where the file will be saved.

        Raises:
            ValueError: If the filename does not contain a valid year.

        Returns:
            FileMetadata: An instance with extracted metadata.
        """
        if year_match := re.search(r"d(\d{4})_", filename):
            return cls(
                filename=filename,
                year=int(year_match.group(1)),
                url=urljoin(base_url, filename),
                output_path=output_dir / filename,
            )
        raise ValueError(f"Invalid filename format: {filename}")


class StormDataProcessor:
    """Process NOAA Storm Events data with asynchronous downloads and efficient processing."""

    def __init__(
        self,
        base_url: URL,
        output_dir: FilePath = "storm_data",
        start_year: Year | None = None,
        end_year: Year | None = None,
        chunk_size: int = 8192,
        max_concurrent_downloads: int = 5,
    ) -> None:
        """
        Initialize the StormDataProcessor.

        Args:
            base_url (URL): Base URL for NOAA Storm Events data
            output_dir (FilePath): Directory to store downloaded files
            start_year (int): Starting year for data collection (inclusive)
            end_year (int): Ending year for data collection (inclusive)
            chunk_size (int): Size of chunks when downloading files
            max_concurrent_downloads (int): Maximum number of concurrent downloads
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.date_range = (
            range(start_year, end_year + 1) if start_year and end_year else None
        )
        self.chunk_size = chunk_size
        self.max_concurrent_downloads = max_concurrent_downloads
        self.processed_files: list[Path] = []

        # Setup logging
        self._configure_logging()

        # Initialize directory structure
        self.directories = self._setup_directories()

    @staticmethod
    def _configure_logging() -> None:
        """Configure logging to both file and console with rotating file handler."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("storm_data_processing.log"),
                logging.StreamHandler(),
            ],
        )

    def _setup_directories(self) -> dict[DirectoryType, Path]:
        """
        Create and return the directory structure for data processing.

        Returns:
            Dictionary mapping directory types to Path objects
        """
        directories = {
            DirectoryType.RAW: self.output_dir / "raw",
            DirectoryType.EXTRACTED: self.output_dir / "extracted",
            DirectoryType.COMBINED: self.output_dir / "combined",
        }

        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories

    def _is_within_date_range(self, year: Year) -> bool:
        """
        Check if the given year falls within the configured date range.

        Args:
            year (Year): The year to check.

        Returns:
            bool: True if within range, False otherwise.
        """
        return True if self.date_range is None else year in self.date_range

    async def _get_file_list(self) -> list[FileMetadata]:
        """
        Retrieve and parse the list of available files from the base URL.

        Returns:
            List of FileMetadata objects for available files

        Raises:
            aiohttp.ClientError: If the HTTP request fails
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url) as response:
                response.raise_for_status()
                content = await response.text()

        pattern = re.compile(r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d{8}\.csv\.gz")
        filenames = sorted(set(pattern.findall(content)))

        metadata_list = [
            FileMetadata.from_filename(
                filename=filename,
                base_url=self.base_url,
                output_dir=self.directories[DirectoryType.RAW],
            )
            for filename in filenames
            if (match := re.search(r"d(\d{4})_", filename))
            and self._is_within_date_range(int(match.group(1)))
        ]

        logging.info(f"Found {len(metadata_list)} files within date range")
        return metadata_list

    async def _download_file(
        self,
        metadata: FileMetadata,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        """
        Download a single file with progress tracking.

        Args:
            metadata (FileMetadata): FileMetadata object containing file information.
            session (aiohttp.ClientSession): aiohttp ClientSession for HTTP requests.
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads.

        Returns:
            bool: True if download was successful, False otherwise.
        """
        if metadata.output_path.exists():
            logging.info(f"File already exists: {metadata.filename}")
            self.processed_files.append(metadata.output_path)
            return True

        async with semaphore:
            try:
                async with session.get(metadata.url) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("Content-Length", 0) or 0)
                    if not total_size:
                        logging.warning(
                            f"Could not determine content length for {metadata.filename}"
                        )

                    async with aiofiles.open(metadata.output_path, "wb") as f:
                        async for chunk in tqdm(
                            response.content.iter_chunked(self.chunk_size),
                            total=total_size // self.chunk_size if total_size else None,
                            desc=f"Downloading {metadata.filename}",
                            unit="chunk",
                        ):
                            await f.write(cast(bytes, chunk))

                # If file is gzipped, extract it
                if metadata.filename.endswith(".gz"):
                    await self._extract_gzip_file(metadata.output_path)

                self.processed_files.append(metadata.output_path)
                logging.info(
                    f"Successfully downloaded and processed: {metadata.filename}"
                )
                return True

            except aiohttp.ClientResponseError as e:
                logging.error(
                    f"HTTP error downloading {metadata.filename}: {e}", exc_info=True
                )
                return False
            except Exception as e:
                logging.error(
                    f"Error downloading {metadata.filename}: {e}", exc_info=True
                )
                return False

    async def _extract_gzip_file(self, gz_path: Path) -> None:
        """
        Extract a gzipped file asynchronously.

        Args:
            gz_path (Path): Path to the gzipped file.

        Raises:
            OSError: If file extraction fails.
        """
        output_path = self.directories[DirectoryType.EXTRACTED] / gz_path.stem
        try:
            with gzip.open(gz_path, "rb") as f_in, output_path.open("wb") as f_out:
                await asyncio.to_thread(shutil.copyfileobj, f_in, f_out)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise OSError("Extracted file is empty or not created")
            logging.info(f"Successfully extracted: {gz_path.name}")
        except Exception as e:
            logging.error(f"Error extracting {gz_path.name}: {e}")
            raise

    def _process_csv_file(self, csv_file: Path) -> pd.DataFrame:
        """
        Process a single CSV file and return a DataFrame with metadata.

        Args:
            csv_file (Path): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed DataFrame with added metadata.
        """
        try:
            df = pd.read_csv(csv_file, low_memory=False, chunksize=100_000).__next__()
            df["source_file"] = csv_file.name
            if year_match := re.search(r"d(\d{4})_", csv_file.name):
                df["data_year"] = int(year_match.group(1))
            return df
        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {e}")
            return pd.DataFrame()

    def combine_csv_files(self) -> Path | None:
        """
        Combine all extracted CSV files into a single dataset with metadata.

        Returns:
            Path | None: Path to the combined CSV file, or None if combination fails.
        """
        logging.info("Starting CSV combination process...")
        csv_files = list(self.directories[DirectoryType.EXTRACTED].glob("*.csv"))
        if not csv_files:
            logging.warning("No CSV files found to combine")
            return None

        try:
            with ThreadPoolExecutor() as executor:
                dfs = list(executor.map(self._process_csv_file, csv_files))
            dfs = [df for df in dfs if not df.empty]

        except Exception as e:
            logging.error(f"Error processing CSV files: {e}")
            return None

        if not dfs:
            logging.error("No data frames created – combination failed")
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        date_range_str = (
            f"{self.date_range.start}-{self.date_range.stop - 1}"
            if self.date_range
            else "all"
        )
        output_file = (
            self.directories[DirectoryType.COMBINED]
            / f"combined_storm_events_{date_range_str}.csv"
        )

        try:
            combined_df.to_csv(output_file, index=False)
            logging.info(f"Successfully created combined file: {output_file}")
            logging.info(f"Combined {len(dfs)} files into {len(combined_df)} records")
        except Exception as e:
            logging.error(f"Error saving combined CSV file: {e}")
            return None

        return output_file

    async def process_files(self) -> None:
        """
        Download and process all storm event files asynchronously using a Task Group.
        """
        file_list = await self._get_file_list()
        if not file_list:
            logging.warning("No files found matching the specified criteria")
            return

        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with aiohttp.ClientSession() as session:
            tasks: list[asyncio.Task] = []
            async with asyncio.TaskGroup() as tg:
                for metadata in file_list:
                    tasks.append(
                        tg.create_task(
                            self._download_file(metadata, session, semaphore)
                        )
                    )
            # All tasks in the group have now completed.
            successful = sum(1 for task in tasks if task.result() is True)
            failed = len(tasks) - successful
            logging.info(f"Download complete. Success: {successful}, Failed: {failed}")


async def main() -> None:
    """Main entry point for the storm data processing application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and process NOAA Storm Events data."
    )
    parser.add_argument(
        "--start-year", type=int, help="Starting year for data collection"
    )
    parser.add_argument("--end-year", type=int, help="Ending year for data collection")
    parser.add_argument(
        "--output-dir", default="storm_data", help="Output directory for data files"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent downloads",
    )
    args = parser.parse_args()

    if args.start_year and args.end_year and args.start_year > args.end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    processor = StormDataProcessor(
        base_url=base_url,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        max_concurrent_downloads=args.max_workers,
    )

    await processor.process_files()
    combined_file = processor.combine_csv_files()
    if combined_file:
        print(f"\nProcessing complete! Combined file created at: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
