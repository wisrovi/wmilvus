"""CLI interface for WMilvus database management, inspection, backup, and restore."""

import json
import click
from pymilvus import MilvusClient


@click.group()
def main() -> None:
    """WMilvus CLI - Manage Milvus Vector Database."""
    pass


@main.command()
@click.option("--uri", default="http://localhost:19530", help="Milvus server URI")
def ping(uri: str) -> None:
    """Ping a Milvus server to test connectivity."""
    click.echo(f"Pinging Milvus at {uri}...")
    try:
        client = MilvusClient(uri=uri)
        collections = client.list_collections()
        click.echo(f"Successfully connected to Milvus! Collections: {collections}")
    except Exception as e:
        click.echo(f"Failed to connect to Milvus: {e}", err=True)


@main.command()
@click.option("--uri", default="http://localhost:19530", help="Milvus server URI")
@click.argument("collection")
def inspect(uri: str, collection: str) -> None:
    """Inspect collection schema, field parameters, and record statistics."""
    try:
        client = MilvusClient(uri=uri)
        if not client.has_collection(collection_name=collection):
            click.echo(f"Collection '{collection}' does not exist.", err=True)
            return

        desc = client.describe_collection(collection_name=collection)
        stats = client.get_collection_stats(collection_name=collection)
        click.echo(f"=== Collection Info: {collection} ===")
        click.echo(json.dumps(desc, indent=2))
        click.echo("=== Statistics ===")
        click.echo(json.dumps(stats, indent=2))
    except Exception as e:
        click.echo(f"Error inspecting collection '{collection}': {e}", err=True)


if __name__ == "__main__":
    main()
