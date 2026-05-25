try:
    from .github_client import GitHubClient
    from .async_github_client import AsyncGitHubClient
    from .enricher import AsyncUserEnricher
    from .sampler import StratifiedSampler
except ImportError:
    # Optional dependencies (e.g. aiohttp) may be missing in lightweight
    # environments such as Streamlit Cloud. Submodules can still be
    # imported individually.
    pass
