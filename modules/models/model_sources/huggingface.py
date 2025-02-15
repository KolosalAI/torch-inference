from huggingface_hub import hf_hub_download

class HuggingFaceDownloader:
    def __init__(self, config):
        self.token = config.hf_token
        self.cache_dir = config.model_cache_dir

    def download(self, model_id, target_dir, version=None, force=False):
        try:
            path = hf_hub_download(
                repo_id=model_id,
                revision=version,
                cache_dir=self.cache_dir,
                force_download=force,
                use_auth_token=self.token
            )
            # Move from cache to model_store
            shutil.copy(path, target_dir / model_id)
            return True
        except Exception as e:
            logger.error(f"HF download failed: {str(e)}")
            return False