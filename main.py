import sys
import click

@click.group()
@click.version_option()
def cli():
    """Witch: play hearthstone using AI black magic."""


@click.command(help="To train the model")
def train():
    pass


cli.add_command(train)


if __name__ == '__main__':
    cli()
