"""CLI for WMilvus."""

import click


@click.group()
def main() -> None:
    """WMilvus CLI - Manage Milvus Vector Database."""
    pass


@main.command()
@click.option("--host", default="localhost", help="Milvus host")
@click.option("--port", default=19530, help="Milvus port")
def ping(host: str, port: int) -> None:
    """Ping a Milvus server to test connectivity."""
    click.echo(f"Pinging Milvus at {host}:{port}...")


if __name__ == "__main__":
    main()
