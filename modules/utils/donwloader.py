import requests
from tqdm import tqdm
import os

class DownloadHandler:
    @staticmethod
    def download_file(url, destination, checksum=None):
        """Generic file downloader with progress tracking"""
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(destination, 'wb') as f:
                    with tqdm(
                        total=total_size, unit='B', 
                        unit_scale=True, desc=url.split('/')[-1]
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            if checksum and not self.verify_checksum(destination, checksum):
                os.remove(destination)
                return False
                
            return True
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False