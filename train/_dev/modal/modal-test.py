# pip install modal
# python -m modal setup
# Writes token to: /home/lhl/.modal.toml
# modal run modal-test.py

import modal

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
