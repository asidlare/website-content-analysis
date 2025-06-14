import asyncio
import typer
from app.api.embeddings import create_embeddings
from app.api.nouns import calculate_frequencies_and_save_to_csv
from app.api.similarities import save_similarities_to_csv

app = typer.Typer()


@app.command()
def create_embeddings_collections() -> None:
    typer.echo("Starting calculations...")
    asyncio.run(create_embeddings())
    typer.echo("Done!")


@app.command()
def calculate_and_save_similarities() -> None:
    typer.echo("Starting calculations...")
    save_similarities_to_csv()
    typer.echo("Done!")


@app.command()
def calculate_and_save_frequencies():
    typer.echo("Starting calculations...")
    calculate_frequencies_and_save_to_csv()
    typer.echo("Done!")


if __name__ == "__main__":
    app()
