# utils/cache_manager.py
"""Cache management utilities for development work."""

import logging
from pathlib import Path
from typing import Dict, Any
import click

from .llm_cache import llm_cache, agent_cache
from .cache import vlm_cache

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache management for all cache types in the application."""

    def __init__(self):
        self.llm_cache = llm_cache
        self.agent_cache = agent_cache
        self.vlm_cache = vlm_cache

    def get_all_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache types."""
        return {
            "llm_cache": self.llm_cache.get_cache_stats(),
            "agent_cache": self.agent_cache.get_cache_stats(),
            "vlm_cache": {
                "memory_cache_size": len(self.vlm_cache.memory_cache),
                "disk_cache_size": len(
                    list(Path(self.vlm_cache.cache_dir).glob("*.pkl"))
                ),
                "max_size": self.vlm_cache.max_size,
                "cache_directory": str(self.vlm_cache.cache_dir),
            },
        }

    def clear_all_caches(self) -> Dict[str, int]:
        """Clear all caches and return counts of cleared items."""
        results = {}

        # Clear LLM cache
        llm_stats = self.llm_cache.get_cache_stats()
        total_llm_items = llm_stats["memory_cache_size"] + llm_stats["disk_cache_size"]
        self.llm_cache.clear_all()
        results["llm_cache_cleared"] = total_llm_items

        # Clear agent cache
        agent_stats = self.agent_cache.get_cache_stats()
        total_agent_items = (
            agent_stats["memory_cache_size"] + agent_stats["disk_cache_size"]
        )
        self.agent_cache.clear_all()
        results["agent_cache_cleared"] = total_agent_items

        # Clear VLM cache (memory only, no clear_all method)
        vlm_memory_count = len(self.vlm_cache.memory_cache)
        self.vlm_cache.memory_cache.clear()
        results["vlm_cache_cleared"] = vlm_memory_count

        total_cleared = sum(results.values())
        results["total_cleared"] = total_cleared

        logger.info(f"[CHE] Cleared all caches: {total_cleared} items")
        return results

    def clear_expired_caches(self) -> Dict[str, int]:
        """Clear only expired cache entries."""
        results = {}

        # Clear expired LLM cache
        results["llm_expired_cleared"] = self.llm_cache.clear_expired()

        # Clear expired agent cache
        results["agent_expired_cleared"] = self.agent_cache.clear_expired()

        # VLM cache doesn't have expiration, skip
        results["vlm_expired_cleared"] = 0

        total_cleared = sum(results.values())
        results["total_expired_cleared"] = total_cleared

        logger.info(f"[CHE] Cleared expired: {total_cleared} items")
        return results

    def log_cache_report(self):
        """Log a detailed cache report using standardized logging."""
        stats = self.get_all_cache_stats()

        logger.info("[CHE] Cache Report:")

        # LLM Cache
        llm_stats = stats["llm_cache"]
        logger.info(
            f"[CHE] LLM: {llm_stats['memory_cache_size']}mem + {llm_stats['disk_cache_size']}disk items"
        )

        # Agent Cache
        agent_stats = stats["agent_cache"]
        logger.info(
            f"[CHE] Agent: {agent_stats['memory_cache_size']}mem + {agent_stats['disk_cache_size']}disk items"
        )

        # VLM Cache
        vlm_stats = stats["vlm_cache"]
        logger.info(
            f"[CHE] VLM: {vlm_stats['memory_cache_size']}mem + {vlm_stats['disk_cache_size']}disk items"
        )

    def print_cache_report(self):
        """DEPRECATED: Use log_cache_report() instead."""
        logger.warning(
            "[DEPRECATED] print_cache_report() is deprecated. Use log_cache_report() instead."
        )
        self.log_cache_report()

        stats = self.get_all_cache_stats()
        llm_stats = stats["llm_cache"]
        agent_stats = stats["agent_cache"]
        vlm_stats = stats["vlm_cache"]

        print(f"  Disk: {vlm_stats['disk_cache_size']} items")
        print(f"  Max Size: {vlm_stats['max_size']}")
        print(f"  Directory: {vlm_stats['cache_directory']}")

        # Totals
        total_memory = (
            llm_stats["memory_cache_size"]
            + agent_stats["memory_cache_size"]
            + vlm_stats["memory_cache_size"]
        )
        total_disk = (
            llm_stats["disk_cache_size"]
            + agent_stats["disk_cache_size"]
            + vlm_stats["disk_cache_size"]
        )

        print("\nTotal Cached Items:")
        print(f"  Memory: {total_memory}")
        print(f"  Disk: {total_disk}")
        print(f"  Grand Total: {total_memory + total_disk}")
        print("==================\n")


# Global cache manager instance
cache_manager = CacheManager()


# CLI Commands for cache management
@click.group()
def cache_cli():
    """Cache management commands."""
    pass


@cache_cli.command()
def stats():
    """Show cache statistics."""
    cache_manager.print_cache_report()


@cache_cli.command()
def clear_all():
    """Clear all caches."""
    click.echo("Clearing all caches...")
    results = cache_manager.clear_all_caches()
    click.echo(f"Cleared {results['total_cleared']} total cache items:")
    click.echo(f"  LLM Cache: {results['llm_cache_cleared']} items")
    click.echo(f"  Agent Cache: {results['agent_cache_cleared']} items")
    click.echo(f"  VLM Cache: {results['vlm_cache_cleared']} items")


@cache_cli.command()
def clear_expired():
    """Clear only expired cache entries."""
    click.echo("Clearing expired cache entries...")
    results = cache_manager.clear_expired_caches()
    click.echo(f"Cleared {results['total_expired_cleared']} expired cache items:")
    click.echo(f"  LLM Cache: {results['llm_expired_cleared']} items")
    click.echo(f"  Agent Cache: {results['agent_expired_cleared']} items")


@cache_cli.command()
@click.argument("cache_type", type=click.Choice(["llm", "agent", "vlm", "all"]))
def clear(cache_type):
    """Clear specific cache type."""
    if cache_type == "all":
        results = cache_manager.clear_all_caches()
        click.echo(f"Cleared all caches: {results['total_cleared']} items")
    elif cache_type == "llm":
        cache_manager.llm_cache.clear_all()
        click.echo("Cleared LLM cache")
    elif cache_type == "agent":
        cache_manager.agent_cache.clear_all()
        click.echo("Cleared Agent cache")
    elif cache_type == "vlm":
        cache_manager.vlm_cache.memory_cache.clear()
        click.echo("Cleared VLM cache")


if __name__ == "__main__":
    # Enable direct execution for development
    import sys

    if len(sys.argv) > 1:
        cache_cli()
    else:
        cache_manager.print_cache_report()
