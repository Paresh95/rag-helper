import os
from litellm import completion


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in your environment.")

    resp = completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )

    print("Success! Output:")
    print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
