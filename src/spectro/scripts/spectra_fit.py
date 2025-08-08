# currently testing typer
import typer


# def hi(name: str):
#     print("hello: ", name)
#
#
# def main():
#     typer.run(hi)
#

app = typer.Typer()

@app.command()
def hi(name: str):
    print("hi: ", name)
